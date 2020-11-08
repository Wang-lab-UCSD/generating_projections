import Bio, numpy as np, torch, os
from Bio import PDB
from projgen_class import ProjGen
import pickle


def main():
    os.chdir("selected_pdb_structs")
    file_list = [filename for filename in os.listdir() if filename.endswith(".ent")]
    proj_dict = dict()
    for i, filename in enumerate(file_list):
        coords, struct_name = ProjGen.get_onehundred_coords(filename)
        proj = ProjGen.project_coords(coords)
        proj_dict[struct_name] = {"original_coordinates":coords, "projections":proj}
        if i % 100 == 0:
            print(i)
    with open("projections.pk", "wb") as outhandle:
        pickle.dump(proj_dict, outhandle)




if __name__ == "__main__":
    main()
