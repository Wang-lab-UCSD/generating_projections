import numpy as np, sys
import Bio
from Bio import PDB

NUM_PROJECTIONS = 100

class ProjGen():

    @staticmethod
    def get_onehundred_coords(filename):
        parser = PDB.PDBParser()
        struct_name = filename.split('pdb')[1].split('.ent')[0]
        struct = parser.get_structure(struct_name, filename)
        coords_list = []
        for model in struct:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        coords_list.append(atom.get_coord())

        coords = np.stack(coords_list)
        indices = np.random.choice(coords.shape[0], 100, replace=False)
        #We are zero-centering the coordinates. Not strictly necessary but also
        #not a bad idea.
        coords = coords[indices,:]
        return coords - np.mean(coords, axis=0), struct_name


    def project_coords(coords):
        phi = np.random.uniform(0,2*np.pi,size=(NUM_PROJECTIONS))
        psi = np.random.uniform(0,2*np.pi,size=(NUM_PROJECTIONS))

        univec1, univec2 = np.zeros((NUM_PROJECTIONS,3)), np.zeros((NUM_PROJECTIONS,3))
        
        univec1[:,0] = np.sin(phi)*np.cos(psi)
        univec1[:,1] = np.sin(phi)*np.sin(psi)
        univec1[:,2] = np.cos(phi)

        univec2[:,0] = np.cos(phi)*np.cos(psi)
        univec2[:,1] = np.cos(phi)*np.sin(psi)
        univec2[:,2] = -np.sin(phi)
        
        proj = np.zeros((NUM_PROJECTIONS,2,NUM_PROJECTIONS))
        proj[:,0,:] = np.dot(coords, univec1.T)
        proj[:,1,:] = np.dot(coords, univec2.T)
        
        if np.allclose(np.sum(univec1*univec2, axis=1), 
                            np.zeros((NUM_PROJECTIONS))) == False:
            print("ERROR! Orthogonal projection vectors not generated correctly!")
            sys.exit()
        
        vecnorms1, vecnorms2 = np.linalg.norm(univec1, axis=1), np.linalg.norm(univec2, axis=1)
        if np.allclose(vecnorms1, np.ones((NUM_PROJECTIONS))) == False or np.allclose(vecnorms2, 
                                np.ones((NUM_PROJECTIONS))) == False:
            print("ERROR! Unit vectors not correctly generated!")
            sys.exit()

        return proj
