# partfind
 
 Author Dr T Hazlehurst, University of Leeds
 
 This is partfind v2
 
 To use:
 from partfind import PartFind
 
 
Useful Functions:
.get_vectors(model_list)

model_list: a list of step file models i.e. ['step1.stp','step2.stp']
This returns a dict of step model names with the assocated vectors

.compare_pairs(model_list)

model_list of two models, return distance between to models and also returns model dict as with get_vectors

.find_in_dataset(input_vector,dataset_array,list_length=None)

Sorts model dict dataset into order of closest to furthest from input_vector


args.py
Contains all the model parameters, it also contains "model_loc" which is the trained model that is used for functions described above.
