import torch
from deep_features import *
from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def slicing_loss(image_generated , image_example , example_style = None, use_bcos = True ):
      
      if use_bcos:

        list_activations_generated = resnet18_bcos().get_gram_matrices(image_generated)
        list_activations_example = resnet18_bcos().get_gram_matrices(image_example)

      else:
        list_activations_generated = VGG19().get_gram_matrices(image_generated)
        list_activations_example = VGG19().get_gram_matrices(image_example)

      # iterate over layers
      loss = 0

      for l in range(len(list_activations_example)):

          #loss += torch.nn.MSELoss()(list_activations_example[l] , list_activations_generated[l])
          b = list_activations_example[l].shape[0]
          dim = list_activations_example[l].shape[1]
          n = list_activations_example[l].shape[2]*list_activations_example[l].shape[3]
          # linearize layer activations and duplicate example activations according to scaling factor
          activations_example = list_activations_example[l].view(b, dim, n) #.repeat(1, 1, SCALING_FACTOR*SCALING_FACTOR)
          activations_generated = list_activations_generated[l].view(b, dim, n) #view(b, dim, n*SCALING_FACTOR*SCALING_FACTOR)
          # sample random directions
          U, S, Vh = torch.linalg.svd(activations_example, full_matrices=False)  # SVD on example activations
          directions = Vh[:, :, torch.randperm(Vh.shape[2])[:dim]].to(torch.device("cuda:0")).squeeze()  # Take the first `dim` principal components
          # Normalize directions
          directions = directions / torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))
          # project activations over random directions
          projected_activations_example = torch.einsum('bdn,md->bmn', activations_example, directions)
          projected_activations_generated = torch.einsum('bdn,md->bmn', activations_generated, directions)
          # sort the projections
          sorted_activations_example = torch.sort(projected_activations_example, dim=2)[0]
          sorted_activations_generated = torch.sort(projected_activations_generated, dim=2)[0]
          # L2 over sorted lists
          
          loss += 1000*torch.mean((sorted_activations_example-sorted_activations_generated)**2) #+ \
            #10*torch.norm(get_gram_matrices(list_activations_example[l]) - get_gram_matrices(list_activations_generated[l]) , p = 1)'''

      if example_style is not None:

        for l in range(len(list_activations_example)):
          gram_matrices_ideal = example_style[l]
          gram_matrices_pred =  get_gram_matrices(list_activations_generated[l])
          loss +=  100*((gram_matrices_ideal - gram_matrices_pred) ** 2.).sum()


      return loss