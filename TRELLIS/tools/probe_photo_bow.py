"""Offline CLIP evidence on two automatically cropped photographic ship ends."""
import sys,json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import CLIPModel,CLIPProcessor

root=Path(sys.argv[1]);root.mkdir(exist_ok=False)
photo=Path(sys.argv[2]);im=np.asarray(Image.open(photo).convert('RGBA'))
yy,xx=np.nonzero(im[:,:,3]>128);points=np.c_[xx,yy].astype(float)
center=points.mean(0);_,_,basis=np.linalg.svd(points-center,full_matrices=False)
u=basis[0];u*=1 if u[0]>=0 else -1;v=np.array([-u[1],u[0]])
local=(points-center)@np.stack([u,v],axis=1);lo=local.min(0)-5;hi=local.max(0)+5
mat=np.c_[np.stack([u,v]),-np.stack([u,v])@center-lo]
rgb=im[:,:,:3].copy();rgb[im[:,:,3]<128]=230
aligned=cv2.warpAffine(rgb,mat,(int(hi[0]-lo[0]+1),int(hi[1]-lo[1]+1)),borderValue=(230,230,230))
Image.fromarray(aligned).save(root/'aligned_photo.png')
w=aligned.shape[1]
crops=[Image.fromarray(aligned[:,:int(w*.48)]),Image.fromarray(aligned[:,int(w*.52):])]
for i,p in enumerate(crops):p.save(root/f'end_{i}.png')
modelpath='/root/autodl-tmp/clip_dl/openai_clip-vit-base-patch32'
model=CLIPModel.from_pretrained(modelpath,local_files_only=True).eval()
processor=CLIPProcessor.from_pretrained(modelpath,local_files_only=True)
pairs=[('a photo of the bow of a ship','a photo of the stern of a ship'),
       ('the front end of a ship','the rear end of a ship'),
       ('a close-up of a ship bow and foredeck','a close-up of a ship stern and aft deck'),
       ('the pointed bow of a boat','the transom at the stern of a boat')]
images=crops+[p.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for p in crops]
with torch.no_grad():
 tf=model.get_text_features(**processor(text=[p for pair in pairs for p in pair],return_tensors='pt',padding=True))
 vf=model.get_image_features(**processor(images=images,return_tensors='pt'))
 tf=tf/tf.norm(dim=1,keepdim=True);vf=vf/vf.norm(dim=1,keepdim=True)
 similarity=(vf@tf.T).numpy().reshape(4,len(pairs),2)
 difference=similarity[:,:,0]-similarity[:,:,1]
record={'source':str(photo),'photo_axis_u':u.tolist(),'photo_center':center.tolist(),
        'prompt_pairs':pairs,'cosine_similarities':similarity.tolist(),
        'bow_minus_stern_scores':difference.tolist(),
        'end0_minus_end1_per_prompt':(difference[[0,2]]-difference[[1,3]]).tolist(),
        'note':'uncalibrated evidence; not a semantic accuracy metric'}
(root/'clip_probe.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
