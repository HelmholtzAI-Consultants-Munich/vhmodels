# from vhmodels.models import DinoBloom

# model = DinoBloom()
# model.load_model(variant='s', device='cpu')
# inputs = model.preprocess('path/to/image') # here preproces will be able to take single image, multiple images or path to image/images 
# print(model.embed(inputs, device='cpu'))

# import kipoi

# def main():
#     print("Kipoi loaded successfully!")
#     print(kipoi.get_model("rbp_eclip/XRCC6"))

# if __name__ == "__main__":
#     main()

import vhmodels

model = vhmodels.load_model(project='dinobloom', model='s')

result = model.transform({'inputs': 'vhmodels/models/DinoBloom/001.bmp'})

print(result)