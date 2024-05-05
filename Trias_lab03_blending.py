import cv2
import numpy as np

#Defines the making of the gaussian stack by stacking the images and blurring them after every layer once more
def gaussianstack(image, levels, blur):
    #stacks the image together
    stack = [image]
    #loops through the levels
    for i in range(levels-1):
        #applies gaussian blur to the images iterating how many times to the level
        image = cv2.GaussianBlur(image, (blur, blur), 0)
        print("Applying gaussian blur:", i)

        #appends the image to the stack
        stack.append(image)
    #returns the stack
    return stack

#Laplacian stack is stack created by taking the differences between the levels in the Gaussian stack. 
#the i-th layer of the Laplacian stack is the difference between the i-th and (i + 1)-th layer of the Gaussian stack

#laplacian[i] = gaussian[i] - gaussian[i-1]
#applies the laplacian stack
def laplacianstack(gaussianstack):
    stack = []
    for i in range(len(gaussianstack)-1):
        # Subtract the (i+1)-th layer and i-th layer of the Gaussian stack directly
        laplacian = cv2.subtract(gaussianstack[i], gaussianstack[i+1])
        print("Applying laplacian", i)
        stack.append(laplacian)

    # Appending the last image of the Gaussian stack
    stack.append(gaussianstack[-1])
    return stack


def blend_images(img1, img2, mask, blur):
    #This runs a gaussian stack for both images, having 50 levels 
    img1gaussian = gaussianstack(img1, levels=50, blur=blur)
    img2gaussian = gaussianstack(img2, levels=50, blur=blur)

    #then, this implements the laplacian stack for both images 
    img1laplacian  = laplacianstack(img1gaussian)
    img2laplacian  = laplacianstack(img2gaussian)

    #then, this creates a gaussian stack for the mask, using the levels as the length of the original laplacian stack
    maskgaussian = gaussianstack(mask, levels=len(img1laplacian ), blur=blur)

    # Blend the Laplacian stacks using the weighted masks
    #make new array for blending the laplacian pyramid
    blendedstack  = []
    #iterates through the laplacian stack of the first image
    for i in range(len(img1laplacian)):
        #weight calculation of each image for contribution 
        #so each layer has a calculated weight and normalized for division of 255


        weight = maskgaussian[i].astype(np.float32) / 255.0


        #this is where the blending of the 1st and 2nd image of laplacian occur with the weight calculated before 
        #laplacian layer i of image 1 multiplied by weight 
        #laplacian layer i of image 2 multiplied by the complement of weight of img 1


        blended = img1laplacian [i] * weight + img2laplacian [i] * (1 - weight)
        #they are then added together to combine

        #after that, append it into the blended stack
        blendedstack.append(blended)

    #then, the blended images from the stack are all added together and clipped so that values dont exceed 255
    blended_image = np.clip(np.sum(blendedstack , axis=0), 0, 255).astype(np.uint8)

    # Display the original images and the blended result

    cv2.imshow('Image 1', img1)
    cv2.imshow('Image 2', img2)
    cv2.imwrite('Trias_lab03_blendcrazy.png', blended_image)
    cv2.imshow('Blended Image', blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Sets the blur values
    blur = 15

    # Ask use for file names of the images to be used:
    img1file  = input("Enter the filename for Image 1: ")
    img2file  = input("Enter the filename for Image 2: ")
    maskfile = input("Enter the filename for the mask image: ")

    # Read the images and turn them into grayscale
    img1 = cv2.imread(img1file , cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2file , cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(maskfile, cv2.IMREAD_GRAYSCALE)

    #resize the images so that they are all uniform together 
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))
    mask = cv2.resize(mask, (500, 500))

    # Sets value to black and white, 127 below and then 255 above 
    #x, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Blend the images using Laplacian stacks and Gaussian masks with increased blur
    blend_images(img1, img2, mask, blur)

main()