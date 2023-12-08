import numpy as np
import cv2 as cv
from utils.seg_utils import vis_seg

drawing = False # true if mouse is pressed
ix,iy = -1,-1
def painting_mask(input_mask):
    visual_input = vis_seg(input_mask[0].cpu().numpy())
    radius = 10
    idx_dict = {'0':(0,0,0), '1':(255, 204, 102), '6':(51, 255, 51), '10':(255, 102, 204)}
    color_idx = 0
    def draw_circle(event,x,y,flags,param):
        global ix,iy,drawing
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing == True:
                cv.circle(img,(x,y),radius,idx_dict[str(color_idx)],-1)
                cv.circle(final_mask,(x,y),radius,(color_idx, color_idx, color_idx), -1)
                
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            cv.circle(img,(x,y),radius,idx_dict[str(color_idx)],-1)
            cv.circle(final_mask,(x,y),radius, (color_idx, color_idx, color_idx), -1)
                
    img = visual_input[:,:,::-1].copy()
    final_mask = input_mask[0].unsqueeze(2).repeat(1,1,3).int().cpu().numpy().copy()
    cv.namedWindow('image')
    cv.setMouseCallback('image',draw_circle)

    while(1):
        cv.imshow('image',img)
        k = cv.waitKey(1) & 0xFF
        if k == 61:
            if radius < 20:
                radius += 5
        elif k == 45:
            if radius > 5:
                radius -= 5
        elif k == ord('0'):
            color_idx = 0
        elif k == ord('1'):
            color_idx = 1
        elif k == ord('6'):
            color_idx = 6
        elif k == ord('a'):
            color_idx = 10
        elif k == 27:
            break
    cv.destroyAllWindows()
    return final_mask[:,:,0]
