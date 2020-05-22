#######################################################
#   Author:     Yuan Liu
#   email:      liu1827@purdue.edu
#   ID:         ee364b23
#   Date:       04/10/2019
#######################################################
import os  # List of module import statements
import sys # Each one on a line
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from matplotlib.path import Path
import imageio


def loadTriangles(leftPointFilePath, rightPointFilePath):
    left_list = []
    right_list = []
    left = np.loadtxt(leftPointFilePath)
    right = np.loadtxt(rightPointFilePath)
    tri_left = Delaunay(left)
    # tri_right = Delaunay(right)
    pt_left = left[tri_left.simplices]
    pt_right = right[tri_left.simplices]
    for i in range(len(pt_left)):
        pt_left[i] = np.float64(pt_left[i])
        pt_right[i] = np.float64(pt_right[i])
        l = Triangle(pt_left[i])
        r = Triangle(pt_right[i])
        left_list.append(l)
        right_list.append(r)

    return (left_list,right_list)


class Triangle:
    def __init__(self, vertices):
        if len(vertices) < 3:
            raise ValueError("The passed parameter should have 3 rows")
        for vertex in vertices:
            if len(vertex) < 2:
                raise ValueError("The passed parameter should have 2 columns")
            for element in vertex:
                if type(element) != np.float64:
                    raise ValueError("The elements in the passed parameter should have type of numpy.float64 ")

        self.vertices = vertices

    def getPoints(self):
        #reference: https://stackoverflow.com/questions/21339448/how-to-get-list-of-points-inside-a-polygon-in-python
        x_arr = np.array([self.vertices[0][0], self.vertices[1][0], self.vertices[2][0]])
        y_arr = np.array([self.vertices[0][1], self.vertices[1][1], self.vertices[2][1]])

        rect_x_max = int(np.max(x_arr))
        rect_x_min = int(np.min(x_arr))
        rect_y_max = int(np.max(y_arr))
        rect_y_min = int(np.min(y_arr))

        rect_x,rect_y = np.meshgrid(np.arange(rect_x_min,rect_x_max+1), np.arange(rect_y_min,rect_y_max+1))

        rect_x,rect_y = rect_x.flatten(),rect_y.flatten()
        pts = np.transpose(np.vstack((rect_x,rect_y)))
        p = Path(self.vertices)
        g = p.contains_points(pts)

        # result = []
        # for index in range(len(g)):
        #   if g[index] == True:
        #       result.append(pts[index])
        return np.float64(pts[g])


def trans_mid(Triangle, target):
    matrix_a = np.array([[Triangle.vertices[0][0], Triangle.vertices[0][1], 1 ,0 ,0 ,0],
                         [0, 0, 0, Triangle.vertices[0][0], Triangle.vertices[0][1],1],
                         [Triangle.vertices[1][0], Triangle.vertices[1][1], 1 ,0 ,0 ,0],
                         [0, 0, 0, Triangle.vertices[1][0], Triangle.vertices[1][1],1],
                         [Triangle.vertices[2][0], Triangle.vertices[2][1], 1, 0, 0, 0],
                         [0, 0, 0, Triangle.vertices[2][0], Triangle.vertices[2][1], 1]])

    matrix_b = np.array([[target.vertices[0][0]],
                         [target.vertices[0][1]],
                         [target.vertices[1][0]],
                         [target.vertices[1][1]],
                         [target.vertices[2][0]],
                         [target.vertices[2][1]]])
    matrix_h = np.linalg.solve(matrix_a,matrix_b)

    matrix_h = np.array([[matrix_h[0][0],matrix_h[1][0],matrix_h[2][0]],[matrix_h[3][0],matrix_h[4][0],matrix_h[5][0]],[0,0,1]])

    return matrix_h


class Morpher:

    def __init__(self, leftImage, leftTriangles, rightImage, rightTriangles):

        for triangle in leftTriangles:
            if not isinstance(triangle,Triangle):
                raise TypeError("Passed Triangle in the list has to be an instance of Triangle")

        for triangle in rightTriangles:
            if not isinstance(triangle,Triangle):
                raise TypeError("Passed Triangle in the list has to be an instance of Triangle")

        if type(leftImage) != np.ndarray or type(rightImage) != np.ndarray:
            raise TypeError("Passed Image parameters have to be a type of numpy array")

        if leftImage.dtype != np.uint8 or rightImage.dtype != np.uint8:
            raise TypeError("Passed Image parameters have to be a type of uint8")

        self.leftImage = leftImage
        self.leftTriangles = leftTriangles
        self.rightImage = rightImage
        self.rightTriangles = rightTriangles

    def getImageAtAlpha(self, alpha):
        final = np.zeros(shape=self.leftImage.shape,dtype = np.float64)
        final[0] = (1 - alpha) * self.leftImage[0] + alpha * self.rightImage[0]
        bi_left = RectBivariateSpline(x=np.arange(0, self.leftImage.shape[0]),
                                      y=np.arange(0, self.leftImage.shape[1]), z=self.leftImage)
        bi_right = RectBivariateSpline(x=np.arange(0, self.rightImage.shape[0]),
                                       y=np.arange(0, self.rightImage.shape[1]), z=self.rightImage)

        for i in range(len(self.leftTriangles)):
            tar = (1-alpha) * self.leftTriangles[i].vertices + alpha * self.rightTriangles[i].vertices
            target = Triangle(tar)

            h_left_mid = trans_mid(self.leftTriangles[i],target)
            h_right_mid = trans_mid(self.rightTriangles[i],target)

            #TODO:get points for  target triangles
            midpoint = target.getPoints()
            temp = midpoint.astype(int).T
            temp2 = (np.asmatrix((np.insert(midpoint,2,1,axis = 1)))).T

            # temp2 = np.array([])
            # for point in midpoint:
            #     temp = np.array((point[0],
            #                       point[1],
            #                       1))
            #     temp2 = np.vstack(temp)
            #     temp2 = np.asmatrix(temp2)
            # # print(temp2)

            lp = np.matmul(np.linalg.inv(h_left_mid),temp2)
            rp = np.matmul(np.linalg.inv(h_right_mid),temp2)
            #print(lp,rp)

            final[temp[1],temp[0]] = (1-alpha)*bi_left.ev(lp[1],lp[0]) + alpha*bi_right.ev(rp[1],rp[0])

            # final[int(point[1]),int(point[0])] = (1-alpha)*bi_left.ev(lp[1],lp[0]) + alpha*bi_right.ev(rp[1],rp[0])
            #final = final.astype(np.uint8)

        return final.astype(np.uint8)

    def saveVideo(self, targetFilePath, frameCount, frameRate, includeReserved=True):

        re = []
        temp = imageio.get_writer(targetFilePath,fps=frameRate)
        temp.append_data(self.getImageAtAlpha(0))
        for i in reversed((range(1,frameCount))):
            temp.append_data(self.getImageAtAlpha( 1 / i ))
            if includeReserved == True:
               re.append(self.getImageAtAlpha(1/i))
        re = reversed(re)
        for i in re:
            temp.append_data(i)
        temp.append_data(self.getImageAtAlpha(0))
        temp.close()

class ColorMorpher(Morpher):

    def __init__(self, leftImage, leftTriangles, rightImage, rightTriangles):
        Morpher.__init__(self, leftImage, leftTriangles, rightImage,rightTriangles)

    def getImageAtAlpha(self, alpha):
        final = np.zeros(shape=(self.leftImage.shape[0],self.leftImage.shape[1],3),dtype=np.uint8)
        final[0, :, 0] = (1 - alpha) * self.leftImage[0, :, 0] + alpha * self.rightImage[0, :, 0]
        final[0, :, 1] = (1 - alpha) * self.leftImage[0, :, 1] + alpha * self.rightImage[0, :, 1]
        final[0, :, 2] = (1 - alpha) * self.leftImage[0, :, 2] + alpha * self.rightImage[0, :, 2]

        bi_left_r = RectBivariateSpline(x=np.arange(0, self.leftImage.shape[0]),
                                      y=np.arange(0, self.leftImage.shape[1]), z=self.leftImage[:,:,0], kx=1, ky=1)
        bi_right_r = RectBivariateSpline(x=np.arange(0, self.rightImage.shape[0]),
                                       y=np.arange(0, self.rightImage.shape[1]), z=self.rightImage[:,:,0], kx=1, ky=1)

        bi_left_g = RectBivariateSpline(x=np.arange(0, self.leftImage.shape[0]),
                                      y=np.arange(0, self.leftImage.shape[1]), z=self.leftImage[:,:,1], kx=1, ky=1)
        bi_right_g = RectBivariateSpline(x=np.arange(0, self.rightImage.shape[0]),
                                       y=np.arange(0, self.rightImage.shape[1]), z=self.rightImage[:,:,1], kx=1, ky=1)

        bi_left_b = RectBivariateSpline(x=np.arange(0, self.leftImage.shape[0]),
                                      y=np.arange(0, self.leftImage.shape[1]), z=self.leftImage[:,:,2], kx=1, ky=1)
        bi_right_b = RectBivariateSpline(x=np.arange(0, self.rightImage.shape[0]),
                                       y=np.arange(0, self.rightImage.shape[1]), z=self.rightImage[:,:,2], kx=1, ky=1)

        for i in range(len(self.leftTriangles)):
            tar = (1-alpha) * self.leftTriangles[i].vertices + alpha * self.rightTriangles[i].vertices
            target = Triangle(tar)

            h_left_mid = trans_mid(self.leftTriangles[i],target)
            h_right_mid = trans_mid(self.rightTriangles[i],target)

            #TODO:get points for  target triangles
            midpoint = target.getPoints()
            temp = midpoint.astype(int).T
            temp2 = (np.asmatrix((np.insert(midpoint,2,1,axis = 1)))).T

            # temp2 = np.array([])
            # for point in midpoint:
            #     temp = np.array((point[0],
            #                       point[1],
            #                       1))
            #     temp2 = np.vstack(temp)
            #     temp2 = np.asmatrix(temp2)
            # print(temp2)

            lp = np.matmul(np.linalg.inv(h_left_mid),temp2)
            rp = np.matmul(np.linalg.inv(h_right_mid),temp2)

            final[temp[1], temp[0],0] = (1 - alpha) * bi_left_r.ev(lp[1], lp[0]) + alpha * bi_right_r.ev(rp[1], rp[0])
            final[temp[1], temp[0],1] = (1 - alpha) * bi_left_g.ev(lp[1], lp[0]) + alpha * bi_right_g.ev(rp[1], rp[0])
            final[temp[1], temp[0],2] = (1 - alpha) * bi_left_b.ev(lp[1], lp[0]) + alpha * bi_right_b.ev(rp[1], rp[0])

            final = final.astype(np.uint8)

        return final

if __name__ == '__main__':
    print('answer')

    r = loadTriangles(os.path.join('/home/ecegridfs/a/ee364b23/Documents/labs-YuanLiuLawrence/Lab12','left.txt'),os.path.join('/home/ecegridfs/a/ee364b23/Documents/labs-YuanLiuLawrence/Lab12','right.txt'))
    left_image = np.array(imageio.imread(os.path.join('/home/ecegridfs/a/ee364b23/Documents/labs-YuanLiuLawrence/Lab12','ll.png')))
    right_image = np.array(imageio.imread(os.path.join('/home/ecegridfs/a/ee364b23/Documents/labs-YuanLiuLawrence/Lab12','rr.png')))
    a = ColorMorpher(left_image,r[0],right_image,r[1])
    targetVideoPath = os.path.join('/home/ecegridfs/a/ee364b23/Documents/labs-YuanLiuLawrence/Lab12/TestData', 'GrayMorph.mp4')
    a.saveVideo(targetVideoPath, 20, 5, True)
