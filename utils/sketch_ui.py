import numpy as np
import cv2 as cv
from utils.seg_utils import vis_seg

drawing = False # true if mouse is pressed
ix,iy = -1,-1

def draw_sketch(input_mask):
    visual_input = vis_seg(input_mask[0].cpu().numpy())

    def draw_line(event,x,y,flags,param):
        global ix,iy,drawing
        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing == True:
                cv.line(img,(ix,iy),(x,y),(128,128,128),2)
                cv.line(final_mask,(ix,iy),(x,y),(128,128,128),2)
                ix,iy = x,y
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            cv.line(img,(ix,iy),(x,y),(128,128,128),2)
            cv.line(final_mask,(ix,iy),(x,y),(128,128,128),2)

                
    img = visual_input[:,:,::-1].copy()
    final_mask = np.zeros((512,512,3), np.uint8)
    cv.namedWindow('image')
    cv.setMouseCallback('image',draw_line)

    while(1):
        cv.imshow('image',img)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break
    cv.destroyAllWindows()
    return final_mask[:,:,0], img