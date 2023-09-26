import numpy as np
import pandas as pd
from scipy.stats import chi2


def correct_supporting(vertices):
    # Copy vertices from objects to supporting since the dropdown option was missing in the QA app
    def add_supporting(verts, object_name):
        temp_supporting_vids = []
        supporting_vid = -1
        for id, v_i in enumerate(verts):
            # single key dict
            for k, v in v_i.items():
                if k == object_name:
                    temp_supporting_vids = v
                if k == 'SUPPORTING':
                    supporting_vid = id
        if supporting_vid != -1:
            # append to supporting
            verts[supporting_vid]['SUPPORTING'] += temp_supporting_vids
        return verts

    # correct supporting contacts
    for i, vert in enumerate(vertices):
        for k, v in vert.items():
            if k == 'hot/training/hake_train2015_HICO_train2015_00000019.jpg':
                # copy bicycle contacts to supporting
                v = add_supporting(v, 'BICYCLE')
            if k == 'hot/training/hake_train2015_HICO_train2015_00000020.jpg':
                # copy skateboard contacts to supporting
                v = add_supporting(v, 'SKATEBOARD')
            if k == 'hot/training/hake_train2015_HICO_train2015_00000942':
                # copy bench contacts to supporting
                v = add_supporting(v, 'BENCH')

            # combine all vert_ids into a single list no matter the object
            v = {ki: vi for d in v for ki, vi in d.items()}
            v = [vi for k, vi in v.items()]
            v = [item for sublist in v for item in sublist]
            v = list(set(v))
            # binarize the list to a numpy array
            v_np = np.zeros(6890)
            v_np[v] = 1
            vert[k] = v_np
        vertices[i] = vert
    return vertices

def fleiss_kappa_per_img(vertices):
    """
    Compute Fleiss' kappa per imagename
    Parameters
    ----------
    vertices : list of np arrays where each array is of shape (6890,) and 1 indicates a vertex is selected
    """
    n = len(vertices)  # number of raters
    N = 6890  # number of images
    k = 2  # number of categories

    # compute the observed agreement
    M = np.zeros((N, k))

    for i in range(k):
        M[:, i] = np.sum(vertices == i, axis=0)

    assert np.sum(M) == N * n

    # compute the expected agreement
    p = np.sum(M, axis=0) / (N * n)
    P = (np.sum(M * M, axis=1) - n) / (n * (n - 1))
    Pbar = np.mean(P)
    PbarE = np.sum(p * p)

    # compute Fleiss' kappa
    kappa = (Pbar - PbarE) / (1 - PbarE)
    return kappa

def fleiss_kappa(data):
    """
    Compute Fleiss' kappa per imagename
    Parameters
    ----------
    data : list of dicts where keys are imgnames
    """
    imgnames = sorted(data[0].keys())
    kappas = []
    for img in imgnames:
        kappa_data = []
        for d in data:
            kappa_data.append(d[img])
        kappa_data = np.array(kappa_data)
        kappa_img = fleiss_kappa_per_img(kappa_data)
        print(f'Fleiss\' Kappa for {img}: {kappa_img}')
        kappas.append(kappa_img)

    # computer mean kappa
    kappa = np.mean(kappas)
    return kappa


# Load the combined qa csv file
csv_file = 'quality_assurance_fleiss.csv'
df = pd.read_csv(csv_file)

vertices = df['vertices'].values
vertices = [eval(v) for v in vertices]

vertices = correct_supporting(vertices)

kappa = fleiss_kappa(vertices)

print('Fleiss\' Kappa:', kappa)
