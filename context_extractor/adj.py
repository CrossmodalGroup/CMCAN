import numpy as np
from tqdm import tqdm
import torch

loc = "./DATA/f30k_precomp/"
data_split = "test"

bbox_array = np.load(loc + '%s_ims_bbx.npy' % data_split, allow_pickle=True)
sizes_array = np.load(loc + '%s_ims_size.npy' % data_split, allow_pickle=True)

bbox_len = bbox_array.shape[0]
sizes_len = sizes_array.shape[0]

assert bbox_len == sizes_len

def compute_pseudo(bb_centre):

    K = bb_centre.shape[0]
    bb_centre = torch.tensor(bb_centre)
    # Compute cartesian coordinates (batch_size, K, K, 2)
    pseudo_coord = bb_centre.view( K, 1, 2).contiguous() - \
        bb_centre.view(1, K, 2).contiguous()
    pseudo_coord[:,:,0] = -pseudo_coord[:,:,0]
    # Convert to polar coordinates
    rho = torch.sqrt(
        pseudo_coord[ :, :, 0]**2 + pseudo_coord[ :, :, 1]**2)
    theta = torch.atan2(
        pseudo_coord[ :, :, 1], pseudo_coord[ :, :, 0])
    pseudo_coord = torch.cat(
        (torch.unsqueeze(rho, -1), torch.unsqueeze(theta, -1)), dim=-1)
    return pseudo_coord


def adj_mtx(pseudo_coord, bboxes):

    n_min = 1
    n_region = pseudo_coord.size(0)
    for idx in range(n_region):
        pseudo_coord[idx,idx,:] = 1e9

    pseudo_coord_i = pseudo_coord
    thero = [-3.*np.pi/4., -np.pi/4., np.pi/4., 3.*np.pi/4., np.pi, -np.pi]

    edge_tuples = []
    for i in range(len(thero)-2):

        coord_clone = pseudo_coord_i.clone()
        coord_clone_masked = pseudo_coord_i.clone()

        if i == 3:
            mask_index = ((coord_clone[:,:,1]<thero[i]) | (coord_clone[:,:,1]>thero[i+1])) & ((coord_clone[:,:,1]<thero[i+2]) | (coord_clone[:,:,1]>=thero[0]))
        else:
            mask_index = (coord_clone[:,:,1]<thero[i]) | (coord_clone[:,:,1]>=thero[i+1])

        coord_clone_masked[mask_index] = 1e9

        min_pho = torch.topk(coord_clone_masked[:,:,0],n_min,-1,largest=False)[0]
        min_pho_idx = torch.topk(coord_clone_masked[:,:,0],n_min,-1,largest=False)[1]

        illegal_bool = min_pho>1e3
        edge_tuples_i = []
        for region_i in range(n_region):
            for idx_j in range(n_min):
                if illegal_bool[region_i,idx_j]:
                    pass
                else:
                    edge_tuples_i.append((region_i, min_pho_idx[region_i,idx_j].item()))

        edge_tuples.extend(edge_tuples_i)
    
    edge_tuples = list(set(edge_tuples))

    adj_mtx = torch.zeros(n_region, n_region)
    for rel_idx, rel in enumerate(edge_tuples):
        adj_mtx[rel[0],rel[1]] = 1.
    
    adj_diag = torch.diag(adj_mtx)
    diag_tens = torch.diag_embed(adj_diag)
    adj_mtx = adj_mtx - diag_tens
    adj_mtx = adj_mtx + torch.eye(36)
    
    return adj_mtx


adj_mtxs_list = []
for img_id in tqdm(range(bbox_len), total=bbox_len):
    bboxes = bbox_array[img_id]
    imsize = sizes_array[img_id]

    for i in range(36):
        bbox = bboxes[i]
        bbox[0] /= imsize['image_w']
        bbox[1] /= imsize['image_h']
        bbox[2] /= imsize['image_w']
        bbox[3] /= imsize['image_h']
        bboxes[i] = bbox

    bb_size = (bboxes[:, 2:] - bboxes[ :, :2])
    bb_centre = bboxes[:, :2] + 0.5 * bb_size

    pseudo_coord = compute_pseudo(bb_centre)
    adj_mtxs = adj_mtx(pseudo_coord, bboxes)
    adj_mtxs_list.append(adj_mtxs.numpy())

adj_mtxs = np.stack(adj_mtxs_list, axis=0)
print(adj_mtxs.shape)

np.save(loc + '%s_ims_dir_selfadj4.npy' % data_split, adj_mtxs)
