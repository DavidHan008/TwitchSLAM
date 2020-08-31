import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform

# [[x, y]] => [[x, y, 1]]
def add_ones(x):
    
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def poseRt(R, t):
    pose = np.eye(4)
    pose[0:3, 0:3] = R
    pose[0:3, 3] = t
    
    return pose

def calculate_rt(E):
    """
    E => Essential matrix
    """
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    # svd decomposition
    U, d, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
            
    # calculate rotation matrix
    R = U * W * Vt  # same as np.dot(np.dot(u, W), vt)
    if np.sum(R.diagonal()) < 0.0:
        R = U * W.T * Vt
    # print('R:', R)
    # calculate translation vector
    t = U[:, 2]
    # print('t:', t)
    pose = poseRt(R, t)
    
    return pose

def normalize(Kinv,pts):
        
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]
        
def denormalize(K, point):
    
    target = np.dot(K, np.array([point[0], point[1], 1.0]).T)
    
    # target = target / target[2]
        
    return int(target[0]), int(target[1])

"""
Hyparameters exist
if you want to reconstruct strict image, you should adjust following parmeters
1. Feature Extraction(maxCorners, qualityLevel, minDistance)
2. Ransac Algorithm(min_samples=8, residual_threshold, max_trials)
"""

def extract(img):
    orb_class = cv2.ORB_create()
    # 3d => 2d dimension decomposition
    feat_img = np.mean(img, axis=2).astype(np.uint8)
        
    features = cv2.goodFeaturesToTrack(image=feat_img, maxCorners=2000, qualityLevel=0.01, minDistance=3)
    
    # create keypoints and descriptors
    keypoints = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in features]
    keypoints, descriptors = orb_class.compute(img, keypoints)
        
    return np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints]), descriptors

# match function
def match_frames(frame_prev, frame_curr, Kinv):
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    # match
    results = []
    matches = matcher.knnMatch(frame_curr.des, frame_prev.des, k=2)
    
    print('matches:', len(matches))
    idx1, idx2 = [], []
    
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            p1 = frame_curr.kps[m.queryIdx]
            p2 = frame_prev.kps[m.trainIdx]
            
            if m.distance < 32:
                if m.queryIdx not in idx1 and m.trainIdx not in idx2:
                    # keep around indicies
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    results.append((p1, p2))
    
     
    pose = None
    assert len(results) >= 8           
    target = np.array(results)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
            
    
    # filters
    model, inliners = ransac((target[:, 0], target[:, 1]), FundamentalMatrixTransform, min_samples=8, residual_threshold=0.01, max_trials=200)
    # model, inliners = ransac((target[:, 0], target[:, 1]), EssentialMatrixTransform, min_samples=8, residual_threshold=0.01, max_trials=200)
    results = target[inliners]
            
    # calculate camera pose parameters(R: rotation matrix, t:translation vector)
    pose = calculate_rt(model.params)
    
    return idx1[inliners], idx2[inliners], pose

class Frame(object):
    
    def __init__(self, map_class, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(K)
        IRt = np.eye(4)
        self.pose = IRt  # camera transformation
        # extract
        points, descriptors = extract(img)
        self.des = descriptors
        self.kps = normalize(self.Kinv, points)
        self.pts = [None] * len(self.kps)
        
        self.id = len(map_class.frames)
        map_class.frames.append(self)
        

