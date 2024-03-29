from cgi import test
import model
import extras
import torch
#import load_data
import numpy as np
import constants
import time
import matplotlib.pyplot as plt
def run(epoch=1000, batch_size=64):
    """
    Flow - 
        1.  Load dataset ie images, poses, 
        2.  Find <o> and <d> using poses
        3.  Convert rays to normalized device coordinates
        4.  Divide the rays into batches
            For each batch
                5. Compute x = o +td using stratified sampling
                6. Use positional encoding to compute gamma(x) and gamma(d)
                8. Make batches again
                7. Make predictions using encoded positions
                2. Train model  
    Notes - 
        1. No need to limit <t> in z=o+td when working with ndc
        2. Coarse and fine networks are the same

    """
    torch.cuda.empty_cache()
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    data = np.load('high_res.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    psnrs = []
    
    #print(images.shape)
    num_of_images, H, W= images.shape[0:3]
    testimg, testpose = torch.from_numpy(images[101]).to(device), poses[101]
    #print(testimg.shape)
    images = images[:100,...,:3]
    poses = poses[:100]
    iternums = []
    shape_img = testimg.shape
    #TODO: Load data here using load_data

    rays_o, rays_d = extras.ray(H, W, focal, torch.from_numpy(testpose))
    images = torch.from_numpy(images)
    poses = torch.from_numpy(poses)
    coarse = model.dlss()
    fine = model.nerfdlss()
    
    coarse.to(device)
        
    fine.to(device)
    lst = list(coarse.parameters()) + list(fine.parameters())
    optimizer = torch.optim.Adam(lst, lr=1e-10)
    t = time.time()
    for i in range(epoch):
        coarse.train()
        fine.train()
        img_idx =torch.randint(0, 100, (1,))
        img_target = images[img_idx].to(device)[0]
        pose_target = poses[img_idx, :4, :4].to(device)
        ray_origins, ray_directions = extras.ray(H, W, focal, pose_target)
        ray_origins, ray_directions = ray_origins.to(device), ray_directions.to(device)
        c, f = extras.one_iteration(coarse, fine, ray_origins, ray_directions)
        c,f = c.reshape(shape_img).to(device), f.reshape(shape_img).to(device)
        loss  = torch.nn.functional.mse_loss(c.double(), img_target.double()).double() + torch.nn.functional.mse_loss(f.double(), img_target.double()).double()
        loss.backward()
        psnr = -10 * np.log10(max(loss.item(), 1e-5))
        print('training loss = ', loss)
        optimizer.step()
        optimizer.zero_grad()
        if i%25==0:
            print(i, (time.time() - t) / 25, 'secs per iter')
            t = time.time()
            rgb, rgb2 = extras.one_iteration(coarse, fine, rays_o, rays_d)
            rgb = rgb.reshape(100,100,3).to(device)
            #print(rgb.shape, testimg.shape)
            loss = torch.nn.functional.mse_loss(rgb, testimg)
            psnr = -10 * np.log10(max(loss.item(), 1e-5))
            print('validation loss = ', psnr)


if __name__ == "__main__":
    run()
