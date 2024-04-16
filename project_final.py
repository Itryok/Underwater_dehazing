from PIL import Image,ImageFilter,ImageEnhance,ImageOps
from matplotlib import pyplot as plt
import numpy as np


image1=Image.open('dataset/raw/Img1.png')
imagec1=image1

def plothist(img,str):
    imgR,imgG,imgB=img.split()
    plt.figure(figsize=(20,20))
    plt.suptitle(str)
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Image')
    plt.subplot(122)
    plt.title("Histogram of Image")
    plt.plot(imgR.histogram(),color='red')
    plt.plot(imgG.histogram(),color='green')
    plt.plot(imgB.histogram(),color='blue')
    plt.show()

def chansplit(img):
    imgR,imgG,imgB=img.split()
    x,y=img.size
    Rchan=np.zeros((y,x,3),dtype="uint8")
    Gchan=np.zeros((y,x,3),dtype="uint8")
    Bchan=np.zeros((y,x,3),dtype="uint8")
    #Individual components of image
    Rchan[:,:,0]=imgR
    Gchan[:,:,1]=imgG
    Bchan[:,:,2]=imgB
    #Components to image
    Rchan=Image.fromarray(Rchan)
    Gchan=Image.fromarray(Gchan)
    Bchan=Image.fromarray(Bchan)
    #plot channels
    plt.figure(figsize=(20,20))
    plt.subplot(332)
    plt.title('Original image')
    plt.imshow(img)
    plt.subplot(334)
    plt.title('Red Channel')
    plt.imshow(Rchan)
    plt.subplot(335)
    plt.title('Green Channel')
    plt.imshow(Gchan)
    plt.subplot(336)
    plt.title('Blue Channel')
    plt.imshow(Bchan)
    plt.subplot(338)
    plt.title("Histogram of Image")
    plt.plot(imgR.histogram(),color='red')
    plt.plot(imgG.histogram(),color='green')
    plt.plot(imgB.histogram(),color='blue')
    plt.show()

def RBcompensate(img):
    imgR, imgG, imgB = img.split()
    #min and max pixel values
    minR, maxR = imgR.getextrema()
    minG, maxG = imgG.getextrema()
    minB, maxB = imgB.getextrema()
    #Array
    imgR = np.array(imgR,np.float64)
    imgG = np.array(imgG,np.float64)
    imgB = np.array(imgB,np.float64)
    x,y = img.size
    #normalizing : (0,1)
    for i in range(0, y):
        for j in range(0, x):
            imgR[i][j]=(imgR[i][j]-minR)/(maxR-minR)
            imgG[i][j]=(imgG[i][j]-minG)/(maxG-minG)
            imgB[i][j]=(imgB[i][j]-minB)/(maxB-minB)
    #Mean of channel
    meanR=np.mean(imgR)
    meanG=np.mean(imgG)
    meanB=np.mean(imgB)
    #RB compensation
    for i in range(y):
        for j in range(x):
            imgR[i][j]=int((imgR[i][j]+(meanG-meanR)*(1-imgR[i][j])*imgG[i][j])*maxR)
            imgB[i][j]=int((imgB[i][j]+(meanG-meanB)*(1-imgB[i][j])*imgG[i][j])*maxB)
    # Scale back to og image
    for i in range(0, y):
        for j in range(0, x):
            imgG[i][j]=int(imgG[i][j]*maxG)

    #Compensated image
    compIm = np.zeros((y, x, 3), dtype = "uint8")
    compIm[:, :, 0]= imgR
    compIm[:, :, 1]= imgG
    compIm[:, :, 2]= imgB

    compIm=Image.fromarray(compIm)

    return compIm

def WBgrayworld(img):
    imager, imageg, imageb = img.split()
    #grayscale conversion
    imagegray=img.convert('L') #L=luminance(single channel grayscale mode)
    #Array
    imageR = np.array(imager,np.float64)
    imageG = np.array(imageg,np.float64)
    imageB = np.array(imageb,np.float64)
    imageGray=np.array(imagegray, np.float64)
    x,y = img.size
    #Mean
    meanR=np.mean(imageR)
    meanG=np.mean(imageG)
    meanB=np.mean(imageB)
    meanGray=np.mean(imageGray)

    # Gray World Algorithm
    for i in range(0, y):
        for j in range(0, x):
            imageR[i][j]=int(imageR[i][j]*meanGray/meanR)
            imageG[i][j]=int(imageG[i][j]*meanGray/meanG)
            imageB[i][j]=int(imageB[i][j]*meanGray/meanB)

    #White balanced Image based on RGB
    whitebalancedIm = np.zeros((y, x, 3), dtype = "uint8")
    whitebalancedIm[:, :, 0]= imageR
    whitebalancedIm[:, :, 1]= imageG
    whitebalancedIm[:, :, 2]= imageB

    Img_WB=Image.fromarray(whitebalancedIm)
    return Img_WB

def highpass_filter(image):
    gray_image = ImageOps.grayscale(image)
    blur_radius=5
    blurred_image = gray_image.filter(ImageFilter.GaussianBlur(blur_radius))
    highpass_image = Image.blend(gray_image, blurred_image, alpha=-1)

    # Convert the result back to RGB if needed
    highpass_image_rgb = Image.merge("RGB", [highpass_image, highpass_image, highpass_image])

    return highpass_image_rgb

def sharpen(wbimage,filter):
    if(filter=='gaussian'):
        smoothed_image = wbimage.filter(ImageFilter.GaussianBlur)
    elif(filter=='highboost'):
        gray_image = wbimage.convert("L")
        high_boost_factor = 1.1
        enhancer = ImageEnhance.Contrast(gray_image)
        high_boost_image = enhancer.enhance(high_boost_factor)
        smoothed_image = Image.merge("RGB", [high_boost_image, high_boost_image, high_boost_image])
    elif(filter=='median'):
        smoothed_image = wbimage.filter(ImageFilter.MedianFilter(size=3))
    elif(filter=='highpass'):
        smoothed_image =highpass_filter(wbimage)

    #RGB channel for smoothed image
    smoothedr, smoothedg, smoothedb = smoothed_image.split()

    #RGB channel of whitebalanced image
    imager, imageg, imageb = wbimage.split()

    #To array
    imageR = np.array(imager,np.float64)
    imageG = np.array(imageg,np.float64)
    imageB = np.array(imageb,np.float64)
    smoothedR = np.array(smoothedr,np.float64)
    smoothedG = np.array(smoothedg,np.float64)
    smoothedB = np.array(smoothedb,np.float64)

    x, y=wbimage.size

    #unsharp masking
    for i in range(y):
        for j in range(x):
            imageR[i][j]=2*imageR[i][j]-smoothedR[i][j]
            imageG[i][j]=2*imageG[i][j]-smoothedG[i][j]
            imageB[i][j]=2*imageB[i][j]-smoothedB[i][j]

    #sharpened image
    sharpenIm = np.zeros((y, x, 3), dtype = "uint8")
    sharpenIm[:, :, 0]= imageR
    sharpenIm[:, :, 1]= imageG
    sharpenIm[:, :, 2]= imageB

    return Image.fromarray(sharpenIm)

def hsv_global_equalization(image):
    # Convert to HSV
    hsvimage = image.convert('HSV')

    # Plot HSV Image
    plt.figure(figsize=(20,20))
    plt.subplot(121)
    plt.title("HSV Image")
    plt.imshow(hsvimage)

    Hue, Saturation, Value = hsvimage.split()
    # Perform Equalization on Value Component
    equalizedValue = ImageOps.equalize(Value, mask = None)

    x, y = image.size
    # Create the equalized Image
    equalizedIm = np.zeros((y, x, 3), dtype = "uint8")
    equalizedIm[:, :, 0]= Hue
    equalizedIm[:, :, 1]= Saturation
    equalizedIm[:, :, 2]= equalizedValue
    #Array
    hsvimage = Image.fromarray(equalizedIm, 'HSV')
    # Convert to RGB
    rgbimage = hsvimage.convert('RGB')

    # Plot equalized image
    plt.subplot(1, 2, 2)
    plt.title("Value Equalised RGB Image")
    plt.imshow(rgbimage)

    return rgbimage

def average_fusion(image1, image2,str):
    # Split the images in R, G, B components
    image1r, image1g, image1b = image1.split()
    image2r, image2g, image2b = image2.split()

    # Convert to array
    image1R = np.array(image1r, np.float64)
    image1G = np.array(image1g, np.float64)
    image1B = np.array(image1b, np.float64)
    image2R = np.array(image2r, np.float64)
    image2G = np.array(image2g, np.float64)
    image2B = np.array(image2b, np.float64)

    x, y = image1R.shape

    # Perform fusion by averaging the pixel values
    for i in range(x):
        for j in range(y):
            image1R[i][j]= int((image1R[i][j]+image2R[i][j])/2)
            image1G[i][j]= int((image1G[i][j]+image2G[i][j])/2)
            image1B[i][j]= int((image1B[i][j]+image2B[i][j])/2)

    # Create the fused image
    fusedIm = np.zeros((x, y, 3), dtype = "uint8")
    fusedIm[:, :, 0]= image1R
    fusedIm[:, :, 1]= image1G
    fusedIm[:, :, 2]= image1B

    # Plot the fused image
    plt.figure(figsize=(20,20))
    plt.suptitle(str)
    plt.subplot(1, 3, 1)
    plt.title("Sharpened Image")
    plt.imshow(image1)
    plt.subplot(1, 3, 2)
    plt.title("Value Equalized Image")
    plt.imshow(image2)
    plt.subplot(1, 3, 3)
    plt.title("Average Fused Image")
    plt.imshow(fusedIm)
    plt.show()

    return Image.fromarray(fusedIm)

def pca_fusion(image1, image2,str):
    # Split the images in R, G, B components
    image1r, image1g, image1b = image1.split()
    image2r, image2g, image2b = image2.split()

    # Convert to column vector
    image1R = np.array(image1r, np.float64).flatten()
    image1G = np.array(image1g, np.float64).flatten()
    image1B = np.array(image1b, np.float64).flatten()
    image2R = np.array(image2r, np.float64).flatten()
    image2G = np.array(image2g, np.float64).flatten()
    image2B = np.array(image2b, np.float64).flatten()

    # Get mean of each channel
    mean1R=np.mean(image1R)
    mean1G=np.mean(image1G)
    mean1B=np.mean(image1B)
    mean2R=np.mean(image2R)
    mean2G=np.mean(image2G)
    mean2B=np.mean(image2B)

    # Create a 2*N array where each column represents each image channel
    imageR=np.array((image1R, image2R))
    imageG=np.array((image1G, image2G))
    imageB=np.array((image1B, image2B))

    x, y = imageR.shape

    # Subtract the respective mean from each column
    for i in range(y):
        imageR[0][i]-=mean1R
        imageR[1][i]-=mean2R
        imageG[0][i]-=mean1G
        imageG[1][i]-=mean2G
        imageB[0][i]-=mean1B
        imageB[1][i]-=mean2B

    # Find the covariance matrix
    covR=np.cov(imageR)
    covG=np.cov(imageG)
    covB=np.cov(imageB)

    # Find eigen value and eigen vector
    valueR, vectorR = np.linalg.eig(covR)
    valueG, vectorG = np.linalg.eig(covG)
    valueB, vectorB = np.linalg.eig(covB)

    # Find the coefficients for each channel which will act as weight for images
    if(valueR[0] >= valueR[1]):
        coefR=vectorR[:, 0]/sum(vectorR[:, 0])
    else:
        coefR=vectorR[:, 1]/sum(vectorR[:, 1])

    if(valueG[0] >= valueG[1]):
        coefG=vectorG[:, 0]/sum(vectorG[:, 0])
    else:
        coefG=vectorG[:, 1]/sum(vectorG[:, 1])

    if(valueB[0] >= valueB[1]):
        coefB=vectorB[:, 0]/sum(vectorB[:, 0])
    else:
        coefB=vectorB[:, 1]/sum(vectorB[:, 1])

    # Convert to array
    image1R = np.array(image1r, np.float64)
    image1G = np.array(image1g, np.float64)
    image1B = np.array(image1b, np.float64)
    image2R = np.array(image2r, np.float64)
    image2G = np.array(image2g, np.float64)
    image2B = np.array(image2b, np.float64)

    x, y = image1R.shape

    # Calculate the pixel value for the fused image from the coefficients obtained above
    for i in range(x):
        for j in range(y):
            image1R[i][j]=int(coefR[0]*image1R[i][j]+coefR[1]*image2R[i][j])
            image1G[i][j]=int(coefG[0]*image1G[i][j]+coefG[1]*image2G[i][j])
            image1B[i][j]=int(coefB[0]*image1B[i][j]+coefB[1]*image2B[i][j])

    # Create the fused image
    fusedIm = np.zeros((x, y, 3), dtype = "uint8")
    fusedIm[:, :, 0]= image1R
    fusedIm[:, :, 1]= image1G
    fusedIm[:, :, 2]= image1B

    # Plot the fused image
    plt.figure(figsize=(20,20))
    plt.suptitle(str)
    plt.subplot(1, 3, 1)
    plt.title("Sharpened Image")
    plt.imshow(image1)
    plt.subplot(1, 3, 2)
    plt.title("Value Equalised Image")
    plt.imshow(image2)
    plt.subplot(1, 3, 3)
    plt.title("PCA Fused Image")
    plt.imshow(fusedIm)
    plt.show()

    return Image.fromarray(fusedIm)

chansplit(image1)

#Reb Blue Compensation
imgcomp=RBcompensate(image1)
plt.figure(figsize=(20,20))
plt.subplot(121)
plt.title("Original Image")
plt.imshow(image1)
plt.subplot(122)
plt.title("RB Compensated Image")
plt.imshow(imgcomp)
plt.show()
plothist(imgcomp,"RB Compensated Image")

#Whitebalancing using Grayworld Algorithm
imgwb=WBgrayworld(imgcomp)
plt.figure(figsize=(20,20))
plt.subplot(131)
plt.title("Original Image")
plt.imshow(image1)
plt.subplot(132)
plt.title("RB Compensated Image")
plt.imshow(imgcomp)
plt.subplot(133)
plt.title("White Balanced Image")
plt.imshow(imgwb)
plt.show()

#Sharpening Image
gaussfilImg=sharpen(imgwb,'gaussian')
highboostfilImg=sharpen(imgwb,'highboost')
medianfilImg=sharpen(imgwb,'median')
hpfilImg=sharpen(imgwb,'highpass')
plt.figure(figsize=(20,20))
plt.subplot(231)
plt.title("Original Image")
plt.imshow(image1)
plt.subplot(232)
plt.title("White Balanced Image")
plt.imshow(imgwb)
plt.subplot(233)
plt.title("Sharpened Image(High pass)")
plt.imshow(hpfilImg)
plt.subplot(234)
plt.title("Sharpened Image(Gaussian)")
plt.imshow(gaussfilImg)
plt.subplot(235)
plt.title("Sharpened Image(Highboost)")
plt.imshow(highboostfilImg)
plt.subplot(236)
plt.title("Sharpened Image(Median)")
plt.imshow(medianfilImg)
plt.show()

#HSV_value equalization
valimg=hsv_global_equalization(imgwb)
#Average fusion
gauss_af=average_fusion(gaussfilImg,valimg,"Gaussian Filter")
hb_af=average_fusion(highboostfilImg,valimg,"Highboost Filter")
med_af=average_fusion(medianfilImg,valimg,"Median Filter")
hp_af=average_fusion(hpfilImg,valimg,"Highpass Filter")

#PCA fusion
gauss_pca=pca_fusion(gaussfilImg,valimg,"Gaussian Filter")
hb_pca=pca_fusion(highboostfilImg,valimg,"Highboost Filter")
med_pca=pca_fusion(medianfilImg,valimg,"Median Filter")
hp_pca=pca_fusion(hpfilImg,valimg,"Highpass Filter")

#pca vs af
plt.figure(figsize=(20,20))
plt.subplot(342)
plt.title("Original image")
plt.imshow(image1)
plt.subplot(345)
plt.title("Average fused image(Gaussian)")
plt.imshow(gauss_af)
plt.subplot(349)
plt.title("PCA fused image(Gaussian)")
plt.imshow(gauss_pca)
plt.subplot(346)
plt.title("Average fused image(Highboost)")
plt.imshow(hb_af)
plt.subplot(3,4,10)
plt.title("PCA fused image(Highboost)")
plt.imshow(hb_pca)
plt.subplot(347)
plt.title("Average fused image(Median)")
plt.imshow(med_af)
plt.subplot(3,4,11)
plt.title("PCA fused image(Median)")
plt.imshow(med_pca)
plt.subplot(348)
plt.title("Average fused image(Highpass)")
plt.imshow(hp_af)
plt.subplot(3,4,12)
plt.title("PCA fused image(Highpass)")
plt.imshow(hp_pca)
plt.show()

def psnr(reference, fused, original):
    R2 = np.amax(reference)**2
    MSE = np.sum(np.power(np.subtract(reference, original), 2))
    MSE /= (reference.size[0] * original.size[1])
    PSNR = 10*np.log10(R2/MSE)

    print("Reference vs Original-", "MSE: ", MSE, "PSNR:", PSNR)

    R2 = np.amax(reference)**2
    MSE = np.sum(np.power(np.subtract(reference, fused), 2))
    MSE /= (reference.size[0] * fused.size[1])
    PSNR = 10*np.log10(R2/MSE)
    print("Reference vs Fused   -", "MSE: ", MSE, "PSNR:", PSNR)
    print('')

ref=Image.open('dataset/raw/Img6.png')
print("Using Gaussian filter")
psnr(ref,gauss_af,imagec1)
print("Using Highboost filter")
psnr(ref,hb_af,imagec1)
print("Using Median filter")
psnr(ref,med_af,imagec1)
print("Using Highpass filter")
psnr(ref,hp_af,imagec1)