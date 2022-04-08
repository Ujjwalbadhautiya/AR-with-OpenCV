import cv2
import numpy as np
cap=cv2.VideoCapture(0)
targetimage=cv2.imread("cards.jpg")
video=cv2.VideoCapture("video.mp4")

detection=False
frameCounter=0



targetimage=cv2.resize(targetimage,(640,480))

success,vid_frame= video.read()
height,width,channels=targetimage.shape
vid_frame=cv2.resize(vid_frame,(width,height))

orb=cv2.ORB_create(nfeatures=1000)
key_pnts1,des1=orb.detectAndCompute(targetimage,None)
#targetimage=cv2.drawKeypoints(targetimage,key_pnts1,None)

def stackImages(imgArray,scale,lables=[]):
    sizeW= imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW,sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((sizeH, sizeW, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

while True:
    sucess,webcam=cap.read()
    imgAug=webcam.copy()
    key_pnts2, des2 = orb.detectAndCompute(webcam, None)
    #webcam=cv2.drawKeypoints(webcam,key_pnts2,None)


    if detection ==False:
        video.set(cv2.CAP_PROP_POS_FRAMES,0)
        frameCounter=0
    else:
        if frameCounter==video.get(cv2.CAP_PROP_FRAME_COUNT):
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0
        success,vid_frame=video.read()
        vid_frame = cv2.resize(vid_frame, (width, height))



    bf=cv2.BFMatcher()
    matches=bf.knnMatch(des1,des2,k=2)
    good=[]
    for m,n in matches:
        if m.distance<0.75*n.distance:
            good.append(m)
    print(len(good))
    imgfeatures=cv2.drawMatches(targetimage,key_pnts1,webcam,key_pnts2,good,None,flags=2)

    if len(good)>15:
        src_pts=np.float32([key_pnts1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts=np.float32([key_pnts2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        matrix,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5)
        print(matrix)
        pts=np.float32([[0,0],[0,height],[width,height],[width,0]]).reshape(-1,1,2)
        destination=cv2.perspectiveTransform(pts,matrix)
        img2=cv2.polylines(webcam,[np.int32(destination)],True,(255,0,255),3)
        imgwarp=cv2.warpPerspective(vid_frame,matrix,(webcam.shape[1],webcam.shape[0]))
        #for overlay
        newMask=np.zeros((webcam.shape[0],webcam.shape[1]),np.uint8)
        cv2.fillPoly(newMask,[np.int32(destination)],(255,255,255))
        maskInv=cv2.bitwise_not(newMask)
        imgAug=cv2.bitwise_and(imgAug,imgAug,mask=maskInv)
        imgAug=cv2.bitwise_or(imgwarp,imgAug)
        imgStacked=stackImages(([webcam,vid_frame,targetimage],[imgfeatures,imgwarp,imgAug]),0.5)

    #cv2.imshow("New Mask", imgAug)
    #cv2.imshow("image warp", imgwarp)
    #cv2.imshow("image 2",img2)
    #cv2.imshow("Features Image",imgfeatures)
    #cv2.imshow("Target",targetimage)
    #cv2.imshow("Frame",vid_frame)
    #cv2.imshow("Webcam",webcam)
    cv2.imshow("Webcam",imgStacked)

    cv2.waitKey(1)
    frameCounter+=1