#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
import numpy as np
import nibabel as nib
import mmap
from subprocess import Popen, PIPE, STDOUT
from shutil import rmtree, copyfile, copytree
from tempfile import mkdtemp
import logging
from logging import info, warning, debug
import argparse

from nibabel.processing import resample_to_output, resample_from_to
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.measurements import label
from skimage.exposure import rescale_intensity


def extract_ct_mask(img_src):
    info('calculating lung mask')
    img = nib.as_closest_canonical(img_src)
    img = resample_to_output(img, (2,2,2), order=3)
    im  = img.get_data()

    gim = gaussian_filter(im, 1)
    mask = gim < -300

    mask = binary_fill_holes(mask)

    mask = np.pad(mask, [(3,3), (3,3), (0,0)], mode='constant', constant_values=True)

    labeled, num_features = label(mask)
    mask[labeled == labeled[0,0,0]] = 0

    # remove small islands
    for idx in np.unique(labeled):
        if np.sum(labeled == idx) < 10000:
            mask[labeled==idx] = 0

    # unpad
    mask = mask[3:-3, 3:-3]

    nii_mask = nib.Nifti1Image(mask.astype(np.uint16), img.affine)
    nii_mask = resample_from_to(nii_mask, img_src, order=0)
    return nii_mask


def mask_rescale(img, mask):
    im = img.get_data()
    ma = mask.get_data().astype(np.float)
    im = -im * ma
    im[ma!=0] = rescale_intensity(im[ma!=0], out_range=(0,1))
    return nib.Nifti1Image(im, img.affine)


def call(cmd, stdout=PIPE, stderr=PIPE):
    p = Popen(cmd)
    out, err = p.communicate()
    return out, err


def read_f(fn):
    f = open(fn)
    return mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)


def write_f(f, fn):
    out = open(fn, 'w')
    out.write(f[:])


def image_registration(fixed, moving, fixed_mask=None, moving_mask=None,
                       padding=40, image_domain=(1.0,1.0,1.0)):

    info('resampling to image domain %s mm' % str(image_domain))
    fixed_source  = fixed
    moving_source = moving
    fixed  = resample_to_output(fixed_source,  image_domain, order=3, cval=-1024)
    moving = resample_to_output(moving_source, image_domain, order=3, cval=-1024)

    if fixed_mask == None:
        fixed_mask = extract_ct_mask(fixed)
    else:
        fixed_mask = resample_from_to(fixed_mask, fixed, order=0)

    if moving_mask == None:
        moving_mask = extract_ct_mask(moving)
    else:
        moving_mask = resample_from_to(moving_mask, moving, order=0)

    info('copying files')
    tmp_dir = mkdtemp()
    debug('tmp dir: %s' % tmp_dir)
    fixed.to_filename(path.join(tmp_dir, 'fixed_img-raw.nii.gz'))
    moving.to_filename(path.join(tmp_dir, 'moving_img-raw.nii.gz'))

    info('masking & rescaling')
    fixed = mask_rescale(fixed, fixed_mask)
    fixed.to_filename(path.join(tmp_dir, 'fixed_img.nii.gz'))
    moving  = mask_rescale(moving, moving_mask)
    moving.to_filename(path.join(tmp_dir, 'moving_img.nii.gz'))

    info('padding')
    call(['ImageMath', '3', path.join(tmp_dir, 'fixed_img.nii.gz'),
                'PadImage', path.join(tmp_dir, 'fixed_img.nii.gz'), str(padding)])
    call(['ImageMath', '3', path.join(tmp_dir, 'moving_img.nii.gz'),
                'PadImage', path.join(tmp_dir, 'moving_img.nii.gz'), str(padding)])

    info('image registration')
    logger = logging.getLogger()
    if logger.level <= 10: verbose = '1'
    else: verbose = '0'
    fn_fixed  = path.join(tmp_dir, 'fixed_img.nii.gz')
    fn_moving = path.join(tmp_dir, 'moving_img.nii.gz')
    call(['antsRegistration', '--dimensionality', '3', '--float', '1',
            '--verbose', verbose,
            '--write-composite-transform', '1',
            '--output', '[%s,%s]' % (path.join(tmp_dir, 'transformation_matrix'),
                                     path.join(tmp_dir, 'warped.nii.gz')),
            '--use-histogram-matching', '1',
            '--initial-moving-transform', '[%s,%s,1]' % (fn_fixed, fn_moving),
            '--transform', 'Rigid[0.1]',
                '--metric', 'MI[%s,%s,1,32,Regular,0.25]' % (fn_fixed, fn_moving),
                '--convergence', '[1000x500x250x100]',
                '--shrink-factors', '32x16x8x4',
                '--smoothing-sigmas', '3x2x1x0',
            '--transform', 'Affine[0.1]',
                '--metric', 'MI[%s,%s,1,32,Regular,0.25]' % (fn_fixed, fn_moving),
                '--convergence', '[1000x500x250x100]',
                '--shrink-factors', '32x16x8x4',
                '--smoothing-sigmas', '3x2x1x0',
            '--transform', 'BSplineSyN[0.1,40,0,3]',
                '--metric', 'CC[%s,%s,1,4]' % (fn_fixed, fn_moving),
                '--convergence', '[100x100x100x50]',
                '--shrink-factors', '20x14x7x4',
                '--smoothing-sigmas', '3x2x1x0'])

    info('applying transform')
    call(['antsApplyTransforms', '-d', '3',
              '-i', path.join(tmp_dir, 'moving_img-raw.nii.gz'),
              '-r', path.join(tmp_dir, 'fixed_img.nii.gz'),
              '-t', path.join(tmp_dir,'transformation_matrixComposite.h5'),
              '-o', path.join(tmp_dir, 'warped-raw.nii.gz')])
    #unpad
    call(['ImageMath', '3', path.join(tmp_dir, 'warped-raw.nii.gz'),
                'PadImage', path.join(tmp_dir, 'warped-raw.nii.gz'), str(-padding)])

    warped = nib.load(path.join(tmp_dir, 'warped-raw.nii.gz'))
    warped = resample_from_to(warped, fixed_source, order=3)
    forward = read_f(path.join(tmp_dir, 'transformation_matrixComposite.h5'))
    inverse = read_f(path.join(tmp_dir, 'transformation_matrixInverseComposite.h5'))
    rmtree(tmp_dir)
    return warped, forward, inverse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform image registration for CT lung Images.')
    parser.add_argument('-f', metavar='nifti', type=str, required=True,
                        help='fixed CT image')
    parser.add_argument('-m', metavar='nifti', type=str, required=True,
                        help='moving CT image')
    parser.add_argument('-o', type=str, default='.',
                        help='output directory (default=.)')
    parser.add_argument('-m_f', metavar='nifti', type=str, default=False,
                        help='mask for the fixed CT image (optional)')
    parser.add_argument('-m_m', metavar='nifti', type=str, default=False,
                        help='mask for the moving CT image (optional)')
    parser.add_argument('-v', action='store_true',
                        help='enable verbose mode')
    args = parser.parse_args()

    fixed  = nib.load(args.f)
    moving = nib.load(args.m)
    if args.m_f: fixed_mask = nib.load(args.m_f)
    else: fixed_mask = None
    if args.m_m: moving_mask = nib.load(args.m_m)
    else: moving_mask = None
    if args.v: logging.basicConfig(level=logging.DEBUG)

    cn = path.split(args.f)[-1]
    cnx = cn.split('.nii.gz')[0]
    cnx = cnx.split('.nii')[0]
    warped, forward, inverse = image_registration(fixed, moving, fixed_mask, moving_mask)
    warped.to_filename(path.join(args.o, cn))
    write_f(forward, path.join(args.o, '%s_transformation_matrix_Composite.h5' % cnx))
    write_f(inverse, path.join(args.o, '%s_transformation_matrix_Inverse_Composite.h5' % cnx))
