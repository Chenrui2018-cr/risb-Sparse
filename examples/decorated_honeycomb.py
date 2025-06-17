# ruff: noqa: T201, D100, D103
from itertools import product

import numpy as np

from risb_sparse.embedding import SolveEmbeddingSparse,EmbeddingSparseDummy
from risb_sparse.solve_lattice_ED import LatticeSolver
from risb_sparse.helpers import block_to_full, get_h0_kin_k
from risb_sparse.helpers_triqs import get_h0_loc
from risb_sparse.kweight import SmearingKWeight

# Define kinetic part & interaction
def get_h0_k(tg=0.5, tk=1.0, nkx=25, spin_names=None):
    if spin_names is None:
        spin_names = ["up", "dn"]
    na = 2  # Break up unit cell into 2 clusters
    n_orb = 3  # Number of orbitals/sites per cluster
    phi = 2.0 * np.pi / 3.0  # bloch factor for transforming to trimer orbital basis
    n_k = nkx**2  # total number of k-points

    # Build shifted 2D mesh
    mesh = np.empty(shape=(n_k, 2))
    for idx, coords in enumerate(product(range(nkx), range(nkx))):
        mesh[idx, 0] = coords[0] / nkx + 0.5 / nkx
        mesh[idx, 1] = coords[1] / nkx + 0.5 / nkx

    # Unit cell lattice vectors
    R1 = (3.0 / 2.0, np.sqrt(3.0) / 2.0)
    R2 = (3.0 / 2.0, -np.sqrt(3.0) / 2.0)
    R = np.array((R1, R2)).T

    # Bravais lattice vectors
    G = 2.0 * np.pi * np.linalg.inv(R).T

    # Vectors to inter-triangle nearest neighbors
    d0 = (1.0, 0.0)
    d1 = (-0.5, np.sqrt(3.0) / 2.0)
    d2 = (-0.5, -np.sqrt(3.0) / 2.0)
    d_vec = [d0, d1, d2]

    h0_k = np.zeros([n_k, na, na, n_orb, n_orb], dtype=complex)

    # Construct in inequivalent block matrix structure
    for k, i, j, m, mm in product(
        range(n_k), range(na), range(na), range(n_orb), range(n_orb)
    ):
        kay = np.dot(G, mesh[k, :])

        # Dispersion terms between clusters
        if (i == 0) and (j == 1):
            for a in range(n_orb):
                h0_k[k, i, j, m, mm] += (
                    -(tg / 3.0)
                    * np.exp(1j * kay @ d_vec[a])
                    * np.exp(1j * phi * (mm - m) * a)
                )
        elif (i == 1) and (j == 0):
            for a in range(n_orb):
                h0_k[k, i, j, m, mm] += (
                    -(tg / 3.0)
                    * np.exp(-1j * kay @ d_vec[a])
                    * np.exp(-1j * phi * (m - mm) * a)
                )
        # Local terms on a cluster
        elif (i == j) and (m == mm):
            h0_k[k, i, j, m, mm] = -2.0 * tk * np.cos(m * phi)
        else:
            continue

    # Get rid of the inequivalent block structure
    h0_k_out = {}
    for bl in spin_names:
        h0_k_out[bl] = block_to_full(h0_k)
    return h0_k_out


def get_hubb_trimer(U): # get the hubbard interaction in tetra basis
    n_orb = 3
    
    # compute the U terms under tetra basis
    U_trans = (1/np.sqrt(3))*np.array([
    [1,1,1],
    [1,np.exp(1j*2*np.pi/3),np.exp(-1j*2*np.pi/3)],
    [1,np.exp(1j*4*np.pi/3),np.exp(-1j*4*np.pi/3)]], dtype= complex) # transformation matrix 

    V_AO = np.zeros((3,3,3,3)) # Interaction tensor
    for i in range(n_orb):
        V_AO[i,i,i,i] = U
    V_trimer = np.einsum("ia,jb,kc,ld,ijkl->abcd",U_trans.conj(), U_trans.conj(),U_trans,U_trans,V_AO)
    return V_trimer

#-------------Parameters Setting for the Problem-------------#
# Setup problem and gf_struct for each inequivalent trimer cluster
n_clusters = 2
n_orb = 3
spin_names = ["up", "dn"]

# Setup non-interacting Hamiltonian matrix on the lattice
tg = 0.5
nkx = 20
h0_k = get_h0_k(tg=tg, nkx=nkx, spin_names=spin_names)
V_trimer = get_hubb_trimer(U=2)

# Set up class to work out k-space integration weights
beta = 30  # inverse temperature
n_target = 8  # 2/3rds filling, totally 12 electrons
#n_target = 2 # 1/3 filling per site
kweight = SmearingKWeight(beta=beta, n_target=n_target,method="methfessel-paxton")

# Set up gf_structure of clusters (a list of tuple, where each tuple is (name, n_orb))
gf_struct_molecule = [
    ("up_A", 1),
    ("up_E1", 1),
    ("up_E2", 1),
    ("dn_A", 1),
    ("dn_E1", 1),
    ("dn_E2", 1),
]
gf_struct_molecule_mapping = { # map: {block_proj: block_h0k}
    "up_A": "up",
    "up_E1": "up",
    "up_E2": "up",
    "dn_A": "dn",
    "dn_E1": "dn",
    "dn_E2": "dn",
}
gf_struct = [gf_struct_molecule for _ in range(n_clusters)] # same for all clusters
gf_struct_mapping = [gf_struct_molecule_mapping for _ in range(n_clusters)] # used when h0_k does not have the same block structures as the projectors

# Make projectors onto each trimer cluster
# each cluster has its own projector, with a dict as {block_name: projector}
projectors = [{} for i in range(n_clusters)] 
for i in range(n_clusters):
    projectors[i]["up_A"] = np.eye(n_clusters * n_orb)[0 + i * n_orb : 1 + i * n_orb, :]
    projectors[i]["dn_A"] = np.eye(n_clusters * n_orb)[0 + i * n_orb : 1 + i * n_orb, :]
    projectors[i]["up_E1"] = np.eye(n_clusters * n_orb)[
        1 + i * n_orb : 2 + i * n_orb, :
    ]
    projectors[i]["dn_E1"] = np.eye(n_clusters * n_orb)[
        1 + i * n_orb : 2 + i * n_orb, :
    ]
    projectors[i]["up_E2"] = np.eye(n_clusters * n_orb)[
        2 + i * n_orb : 3 + i * n_orb, :
    ]
    projectors[i]["dn_E2"] = np.eye(n_clusters * n_orb)[
        2 + i * n_orb : 3 + i * n_orb, :
    ]

# Get the non-interacting kinetic Hamiltonian matrix on the lattice
h0_kin_k = get_h0_kin_k(h0_k, projectors, gf_struct_mapping=gf_struct_mapping) # this is the non-local(kinetic) part of the h0_k
#h0_kin_k: {"up":np.array[(nk,nclusters*n_orb,nclusters*n_orb)], "dn":np.array[(nk,nclusters*n_orb,nclusters*n}


# Set up embedding solvers
embedding = [SolveEmbeddingSparse(V_trimer, gf_struct[0])] # construction of the embedding solver needs H_loc and gf_struct giving block structure of the Hamiltonian
for _ in range(n_clusters - 1):
    embedding.append(EmbeddingSparseDummy(embedding[0])) # uniform embedding solver for the rest of clusters``

def symmetries(A):
    n_clusters = len(A)
    # Paramagnetic
    for i in range(n_clusters):
        A[i]["up_A"] = 0.5 * (A[i]["up_A"] + A[i]["dn_A"])
        A[i]["dn_A"] = A[i]["up_A"]
        A[i]["up_E1"] = 0.5 * (A[i]["up_E1"] + A[i]["dn_E1"])
        A[i]["dn_E1"] = A[i]["up_E1"]
        A[i]["up_E2"] = 0.5 * (A[i]["up_E2"] + A[i]["dn_E2"])
        A[i]["dn_E2"] = A[i]["up_E2"]
    
    # E1 = E2.conj()
    for i in range(n_clusters):
        A[i]["up_E2"] = A[i]["up_E1"].conj().T
        A[i]["dn_E2"] = A[i]["dn_E1"].conj().T
    return A

#print("Initial mu:",kweight.mu)
# Setup RISB solver class
S = LatticeSolver(
    h0_k = h0_k,
    gf_struct=gf_struct,
    embedding=embedding,
    update_weights=kweight.update_weights,
    projectors=projectors,
    symmetries=[symmetries],
    gf_struct_mapping=gf_struct_mapping,
)


# Solve
S.solve(tol=1e-4)
# for i in range(5):
#    x = S.solve(one_shot=True)

# Print out some interesting observables
with np.printoptions(formatter={"float": "{: 0.4f}".format}):
    for i in range(S.n_clusters):
        print(f"Cluster {i}:")
        for bl, Z in S.Z[i].items():
            print(f"Quasiaprticle weight Z[{bl}] = \n{Z}")
        for bl, Lambda in S.Lambda[i].items():
            print(f"Correlation potential Lambda[{bl}] = \n{Lambda}")

