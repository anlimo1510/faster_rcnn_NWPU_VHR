# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick edited by fzp
# --------------------------------------------------------

# edit log : done v1.0
# problems exist : [val set] & [test set] confused

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from NWPU_VHR_eval import voc_eval
import annotation_parser_for_NWPU_VHR as ap
from model.config import cfg

import matplotlib.pyplot as plt


class NWPU_VHR(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'NWPU_VHR_'+image_set )
        self._image_set = image_set
        if image_set == 'train':
            self._data_path = os.path.join("/home/fzp/data/NWPU_VHR") # recommend the top directory
        elif image_set == 'test': 
            self._data_path = os.path.join("/home/fzp/data/NWPU_VHR")
        self._classes = ('__background__',
                         'airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 'ground track field', 'harbor', 'bridge', 'vehicle')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes))) 
        self._xml_path = os.path.join(self._data_path, "ground_truth") # annotation set
        self._image_ext = '.jpg' # data extension name
        # load images index for different sets like train or test
        # self._image_index = self._load_xml_filenames()
        self._image_index = self._load_image_set_index() # type = [str]
        self._salt = str(uuid.uuid4()) # unknown func uuid
        self._comp_id = 'comp4'

        # NWPU_VHR specific config options !!!!!!!!! haven't figure it out !!!!!!!!!!
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)


    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_image_filename(self._image_index[i])
 
    def image_path_from_image_filename(self, image_filename):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'positive_image_set',
                                  image_filename + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    # def _load_xml_filenames(self):
    #     """
    #     Load the indexes listed in this dataset's image set file.
    #     """
    #     # Example path to image set file:
    #     xml_folder_path = os.path.join(self._data_path, "Annotations")
    #     assert os.path.exists(xml_folder_path), \
    #         'Path does not exist: {}'.format(xml_folder_path)

    # 	for dirpath, dirnames, filenames in os.walk(xml_folder_path):
    #     	xml_filenames = [xml_filename.split(".")[0] for xml_filename in filenames]

    #     return xml_filenames
    
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path,
                                      'image_set', self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_NWPU_VHR_annotation(xml_filename)
                    for xml_filename in self._image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_NWPU_VHR_annotation(self, xml_filename):
        """
        Load image and bounding boxes info from XML file in the ImageNet format
        """
        filepath = os.path.join( self._xml_path, xml_filename + '.txt')
       

        objects = ap.parse(filepath)

        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objects if int(obj['difficult']) == 0]
            # if len(non_diff_objs) != len(objs):
            #     print 'Removed {} difficult objects'.format(
            #         len(objs) - len(non_diff_objs))
            objects = non_diff_objs

        num_objs = len(objects)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        # Be careful about the starting index 0/1 
        for ix, obj in enumerate(objects):
            box = obj["box"]
            x1 = box['xmin'] 
            y1 = box['ymin'] 
            x2 = box['xmax'] 
            y2 = box['ymax'] 
            # go next if the wnid not exist in declared classes
            try:
                cls = self._class_to_ind[obj["wnid"]]
            except KeyError:
                print "wnid %s isn't show in given"%obj["wnid"]
                continue
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id     
        
    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._data_path,
            'results',
            filename)
        return path     
          
    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} NWPU_VHR detection results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._xml_path,
            '{:s}.txt')
        imagesetfile = os.path.join(
            self._data_path,
            self._image_set + '.txt')
        cachedir = os.path.join(self._data_path, 'annotations_cache')
        aps = []
       
        
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        #color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        #fig, ax = plt.subplots()
        #ax.set_color_cycle(color_cycle)
        
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

           # plt.plot(rec, prec, lw=2, 
           #         label='Precision-recall curve of class {} (area = {:.4f})'
           #         ''.format(cls, ap))

        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.grid(True)
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        # plt.title('Precision-Recall')
        # plt.legend(loc="best")     
        # plt.show()

        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')   
                                       
    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        
        
    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

