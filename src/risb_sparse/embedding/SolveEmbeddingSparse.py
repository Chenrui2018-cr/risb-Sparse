# This file provides an exact diagonalization interface for the RISB Python library:
#     https://thenoursehorse.github.io/risb
#
# The RISB library was developed by:
#     H. L. Nourse and B. J. Powell (2016–2023),
#     R. H. McKenzie (2016–2022).
#
# This interface was written by Chenrui Wang, 2025.
# Copyright (C) 2025 Chenrui Wang
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You may obtain a copy of the License at:
#     https://www.gnu.org/licenses/gpl-3.0.txt


import numpy as np
import time

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from numpy.typing import ArrayLike
from typing import TypeAlias, TypeVar
from itertools import product

GfStructType: TypeAlias = list[tuple[str, int]]
MFType: TypeAlias = dict[ArrayLike]

class SolveEmbeddingSparse:
    """
    Impurity solver of embedding space using user-defined ED solver.
    Parameters
    ----------
    V : numpy array
        Tensor for Interaction Hamiltonian in the embedding space.
    gf_struct : list of pairs [ (str,int), ...]
        Structure of the matrices. It must be a
        list of pairs, each containing the name of the
        matrix block as a string and the size of that block.
        For example: ``[ ('up', 3), ('down', 3) ]``.

    """
    def __init__(self,V:ArrayLike, gf_struct):
        
        #: dict[tuple[str,int]] : Block matrix structure of c-electrons.
        self.gf_struct = gf_struct
        self.L = self.count_orbitals() # including spin 2*number of total orbitals
        self.L_total = 2 * self.L # including impurities
        self.N_particles = int(self.L_total/2) # solve impurity for half-filling
        # preperation for the Exact Diagonalization
        self.basis = self.create_basis(self.L_total, self.N_particles)
        self.index = {state: i for i, state in enumerate(self.basis)}
        self.dim = len(self.basis)
        self.map_dict = self.construct_start_index(gf_struct)
        # Do gf_struct as a map
        self.gf_struct_dict = self._dict_gf_struct(self.gf_struct)

        self.V_int = V # interaction tensor

        # Ground state of the embedding problem
        self.gs_vector = None
        self.gs_energy = None

        #: scipy.sparse.csr_matrix: Embedding Hamiltonian. It is the sum of 
        #: :attr:`h0_loc`,:attr:`h_int`,:attr:`h_hybr`,:attr:`h_bath`
        self.h_emb: csr_matrix = csr_matrix((self.dim, self.dim), dtype=np.complex128)

        #: scipy.sparse.csr_matrix:Single-particle quadratic couplings of
        #: c-electron terms in ::attr:`h_emb`.
        self.h0_loc: csr_matrix = csr_matrix((self.dim, self.dim), dtype=np.complex128)

        #: scipy.sparse.csr_matrix: Interaction couplings of
        #: c-electron terms in ::attr:`h_emb`.
        self.h_int: csr_matrix = csr_matrix((self.dim, self.dim), dtype=np.complex128)

        #: scipy.sparse.csr_matrix: Bath terms in :attr:`h_emb`.
        self.h_bath: csr_matrix = csr_matrix((self.dim, self.dim), dtype=np.complex128)

        #: scipy.sparse.csr_matrix: Hybridization terms in :attr:`h_emb`.
        self.h_hybr: csr_matrix = csr_matrix((self.dim, self.dim), dtype=np.complex128)

        #: dict[numpy.ndarray] : f-electron density matrix.
        self.rho_f = {}

        #: dict[numpy.ndarray] : c-electron density matrix.
        self.rho_c = {}

        #: dict[numpy.ndarray] : Density matrix of hybridization terms
        #: (c- and f-electrons).
        self.rho_cf = {}

    @staticmethod
    def _dict_gf_struct(gf_struct: GfStructType) -> dict[str, int]: # GfStructType -> dict
        return dict(gf_struct)
    
    @staticmethod
    def check_hermiticity(H, atol=1e-6) -> bool:
        """Check if H is Hermitian within tolerance atol."""
        H_dag = H.getH()  # Hermitian conjugate
        diff = (H - H_dag).tocoo()
        max_diff = np.max(np.abs(diff.data)) if diff.nnz > 0 else 0.0
        return max_diff < atol

    # count the number of orbitals for a given gf_struct
    def count_orbitals(self):
    # count the number of particles in a given state
        N = 0
        for bl_name, n_orb in self.gf_struct:
            N += n_orb
        return N


    def count_particles(self, state):
        return bin(state).count('1')
    
    # create the basis of states with N particles in L
    def create_basis(self, L, N):
        return [i for i in range(1 << L) if self.count_particles(i) == N]
    
    # given gf_struct, construct the start index mapping dict to simplify the ED hamiltonian construction
    def construct_start_index(self, gf_struct):
    # return a dict with keys as block names and values as start index j_start
    #  (2(j+j_start) for up and 2(j+j_start)+1 for dn in future calculation)
        map_dict = {}
        idx_up, idx_dn = 0, 0
        for name, n_orb in gf_struct:
            if name.startswith("up"):
                map_dict[name] = idx_up
                idx_up += n_orb
            elif name.startswith("dn"):
                map_dict[name] = idx_dn
                idx_dn += n_orb
            else:
                raise ValueError("spin channel not recognized")
        return map_dict

    def fermion_sign(self, state, i, j): #fermion sign for hopping c_i^dag c_j, return parity of fermions between [i,j-1](i<j) or [j+1,i] (i>j)
        if i == j:
            return 1
        elif i < j:
            return (-1) ** self.count_particles(state & (((1 << j) - 1) & ~((1 << i) - 1)))
        else:
            return (-1) ** self.count_particles(state & (((1 << (i+1)) - 1) & ~((1 << (j+1)) - 1)))
    
    def aux_fermion_sign(self, state, i, j): #fermion sign for hopping f_i f_j^dag, return parity of fermions between 1+[i+1,j](i<j) or [j+1,i] (i>j)
        if i == j:
            return 1
        elif i < j:
            return (-1) ** (self.count_particles(state & (((1 << (j+1)) - 1) & ~((1 << (i+1)) - 1)))+1)
        else:
            return (-1) ** self.count_particles(state & (((1 << (i+1)) - 1) & ~((1 << (j+1)) - 1)))

    def apply_c_dag_c(self, state, i, j):
        if i!= j: # not onsite term
            if not (state & (1 << j)) or (state & (1 << i)):
                return None, 0
            new_state = state ^ (1 << i) ^ (1 << j)
            sign = self.fermion_sign(state, i, j)
        elif i==j:
            if not (state & (1 << i)): # no particle
                return None, 0
            new_state = state 
            sign = 1
        return new_state, sign
    
    def apply_f_f_dag(self,state,i,j): # f_i f_j^dag
        if i!=j:
            if (state &(1<<j)) or not (state & (1<<i)):
                return None, 0
            new_state = state ^ (1<<i) ^ (1<<j)
            sign = self.aux_fermion_sign(state,i,j)
        elif i==j:
            if (state & (1 << i)): # with particle
                return None,0
            new_state = state
            sign = 1
        return new_state, sign

    def apply_c_dag_c_dag_c_c(self, state, i, j, k, l):

        if i==j or k==l: # not occur in our case
            return None, 0
        if not (state & (1 << l)) or not (state & (1 << k)):
            return None, 0
        
        state1 = state ^ (1 << l) ^ (1 << k)
        if k<l:# k<l: [k,l-1]-1
            sign1 = self.fermion_sign(state, k, l)*(-1) # -1 for overcounting of k
        elif k>l: #k>l:[l+1,k]
            sign1 = self.fermion_sign(state, k, l)
        
        if (state1 & (1 << j)) or (state1 & (1 << i)):
            return None, 0
        state2 = state1 ^ (1 << i) ^ (1 << j)
        if i<j: #i<j:1+[i,j-1]
            sign2 = self.fermion_sign(state1, i, j)*(-1)
        elif i>j: #i>j:[j+1,i]
            sign2 = self.fermion_sign(state1, i, j)
        
        return state2, sign1 * sign2
    
    def apply_c_dag_c_c_dag_c(self, state, i, j, k, l):
        # c^\dagger_i c_j c^\dagger_k c_l operator on state
        if j==l or i==k:
            return None, 0
        state1, sign1 = self.apply_c_dag_c(state,k,l)
        if state1 is None:
            return None, 0
        state2, sign2 = self.apply_c_dag_c(state1,i,j)    
        return state2, sign1*sign2
    
    def create_c_dag_c_matrix(self, i: int, j: int) -> csr_matrix:

        r'''
        Create sparse matrix representation of c c^\dagger operator 
        in the current Fock basis.
        '''
        row, col, data = [], [], []
        
        for istate, state in enumerate(self.basis):
            new_state, sign = self.apply_c_dag_c(state, i, j)
            if new_state in self.index:
                jstate = self.index[new_state]
                row.append(jstate)
                col.append(istate)
                data.append(sign)
        op = coo_matrix((data, (row, col)), shape=(self.dim, self.dim), dtype=np.complex128)
        return op.tocsr()
    
    def create_c_c_dag_matrix(self, i: int, j: int) -> csr_matrix:

        r'''
        Create sparse matrix representation of c c^\dagger operator 
        in the current Fock basis.
        '''
        row, col, data = [], [], []
        
        for istate, state in enumerate(self.basis):
            new_state, sign = self.apply_f_f_dag(state, i, j)
            if new_state in self.index:
                jstate = self.index[new_state]
                row.append(jstate)
                col.append(istate)
                data.append(sign)
        
        op = coo_matrix((data, (row, col)), shape=(self.dim, self.dim), dtype=np.complex128)
        return op.tocsr()

    def build_sparse_matrix(self, row, col, data):
        return coo_matrix((data, (row, col)), shape=(self.dim, self.dim)).tocsr()

    def set_h_bath(self, Lambda_c:MFType,test=False) -> None:
        row, col, data = [], [], []
        L = self.L
        for bl, mat in Lambda_c.items():
            assert np.allclose(mat, mat.conj().T), f"{bl} not Hermitian"
            idx0 = self.map_dict[bl]
            for istate, state in enumerate(self.basis):
                for i, j in product(range(mat.shape[0]), repeat=2):
                    if abs(mat[i, j]) < 1e-5:
                        continue
                    if bl.startswith("up"):
                        new_state, sign = self.apply_f_f_dag(state, L + 2 * (i + idx0), L + 2 * (j + idx0))
                    else:
                        new_state, sign = self.apply_f_f_dag(state, L + 2 * (i + idx0) + 1, L + 2 * (j + idx0) + 1)
                    if new_state in self.index:
                        jstate = self.index[new_state]
                        row.append(jstate)
                        col.append(istate)
                        data.append(sign * mat[i, j])

        self.h_bath = self.build_sparse_matrix(row, col, data)
        if test:
            print("Test: solve the h_bath part only")
            import matplotlib.pyplot as plt
            import scipy.sparse
            
            # plot the sparse matrix
            def plot_sparse_matrix(matrix, title="Sparsity Pattern"):
                plt.figure(figsize=(6, 6))
                plt.spy(matrix, markersize=1)
                plt.title(title)
                plt.xlabel("Column index")
                plt.ylabel("Row index")
                plt.grid(False)
                plt.show()

            plot_sparse_matrix(self.h_bath, title="h_bath Sparsity Pattern")

            print("h_bath:",self.h_bath)
            eigenvalue, eigenvector = eigsh(self.h_bath, k=1, which='LM')  # SA: smallest algebraic
            print("GS energy of h_bath:",eigenvalue[0])

    def set_h_hybr(self, D:MFType,test=False) -> None:
        row, col, data = [], [], []
        L = self.L
        for bl, mat in D.items():
            idx0 = self.map_dict[bl]
            for istate, state in enumerate(self.basis):
                for i, j in product(range(mat.shape[0]), repeat=2):
                    if abs(mat[i, j]) < 1e-5:
                        continue
                    if bl.startswith("up"):
                        new_state, sign = self.apply_c_dag_c(state, 2 * (i + idx0), L + 2 * (j + idx0)) # c^dag f
                        new_state_conj, sign_conj = self.apply_c_dag_c(state, L + 2 * (j + idx0), 2 * (i + idx0)) # f^dag c
                    else:
                        new_state, sign = self.apply_c_dag_c(state, 2 * (i + idx0) + 1, L + 2 * (j + idx0) + 1)
                        new_state_conj, sign_conj = self.apply_c_dag_c(state, L + 2 * (j + idx0) + 1, 2 * (i + idx0) + 1)
                    if new_state in self.index:
                        jstate = self.index[new_state]
                        row.append(jstate)
                        col.append(istate)
                        #data.append(mat[i, j])
                        data.append(sign * mat[i, j])
                    if new_state_conj in self.index:
                        jstate_conj = self.index[new_state_conj]
                        row.append(jstate_conj)
                        col.append(istate)
                        #data.append(mat[i, j].conj())
                        data.append(sign_conj * mat[i, j].conj())
        self.h_hybr = self.build_sparse_matrix(row, col, data)
        if test:
            print("Test: solve the h_hybr part only")
            print("h_hybr:",self.h_hybr)
            eigenvalue, eigenvector = eigsh(self.h_hybr, k=1, which='SA')  # SA: smallest algebraic
            print("GS energy of h_hybr:",eigenvalue[0])
    '''
    def set_h_hybr(self, D:MFType,test = True) -> None:
        row, col, data = [], [], []
        L = self.L
        for bl, mat in D.items():
            idx0 = self.map_dict[bl]
            for istate, state in enumerate(self.basis):
                for i, j in product(range(mat.shape[0]), repeat=2):
                    if abs(mat[i, j]) < 1e-5:
                        continue
                    if bl.startswith("up"):
                        new_state, sign = self.apply_c_dag_c(state, 2 * (j + idx0), L + 2 * (i + idx0)) # c^dag f
                        #new_state_conj, sign_conj = self.apply_c_dag_c(state, L + 2 * (i + idx0), 2 * (j + idx0)) # f^dag c
                    else:
                        new_state, sign = self.apply_c_dag_c(state, 2 * (j + idx0) + 1, L + 2 * (i + idx0) + 1)
                        #new_state_conj, sign_conj = self.apply_c_dag_c(state, L + 2 * (i + idx0) + 1, 2 * (j + idx0) + 1)
                    if new_state in self.index:
                        jstate = self.index[new_state]
                        row.append(jstate)
                        col.append(istate)
                        #data.append(mat[i, j])
                        # Add Hermite conjugate term
                        data.append(sign * mat[i, j])
                        row.append(istate)
                        col.append(jstate)
                        data.append(sign * mat[i, j].conj())

        self.h_hybr = self.build_sparse_matrix(row, col, data)
        if test:
            print("Test: solve the h_hybr part only")
            print("h_hybr:",self.h_hybr)
            eigenvalue, eigenvector = eigsh(self.h_hybr, k=1, which='SA')  # SA: smallest algebraic
            print("GS energy of h_hybr:",eigenvalue[0])
        '''
    def set_h0_loc(self, h0_loc_matrix:MFType,test=False) -> None:
        row, col, data = [], [], []
        for bl, mat in h0_loc_matrix.items():
            idx0 = self.map_dict[bl]
            for istate, state in enumerate(self.basis):
                for i, j in product(range(mat.shape[0]), repeat=2):
                    if abs(mat[i, j]) < 1e-5:
                        continue
                    if bl.startswith("up"):
                        new_state, sign = self.apply_c_dag_c(state, 2 * (i + idx0), 2 * (j + idx0))
                    else:
                        new_state, sign = self.apply_c_dag_c(state, 2 * (i + idx0) + 1, 2 * (j + idx0) + 1)
                    if new_state in self.index:
                        jstate = self.index[new_state]
                        row.append(jstate)
                        col.append(istate)
                        data.append(sign * mat[i, j])
        self.h0_loc = self.build_sparse_matrix(row, col, data)
        if test:
            print("Test: solve the h0_loc part only")
            print("h_loc:",self.h0_loc)
            eigenvalue, eigenvector = eigsh(self.h0_loc, k=1, which='SA')  # SA: smallest algebraic
            print("GS energy of h_loc:",eigenvalue[0])

    def set_h_int(self, V,test = False) -> None:
        row, col, data = [], [], []
        for istate, state in enumerate(self.basis):
            for i, j, k, l in product(range(V.shape[0]), repeat=4):
                if abs(V[i, j, k, l]) < 1e-6:
                    continue
                new_state, sign = self.apply_c_dag_c_dag_c_c(state, 2 * i, 2 * j + 1, 2 * k + 1, 2 * l)
                if new_state in self.index:
                    jstate = self.index[new_state]
                    row.append(jstate)
                    col.append(istate)
                    data.append(sign * V[i, j, k, l])
        self.h_int = self.build_sparse_matrix(row, col, data)
        if test:
            print("Test: solve the interaction part only")
            '''
            # Check hermicity
            H = self.h_int.tocsr()
            for r, c, v in zip(H.nonzero()[0], H.nonzero()[1], H.data):
                if abs(v - np.conj(H[c, r])) > 1e-6:
                    print(f"H[{r},{c}] = {v}, H[{c},{r}] = {H[c, r]}")
            '''

            eigenvalue, eigenvector = eigsh(self.h_int, k=1, which='SA')  # SA: smallest algebraic
            print("GS energy of h_int:",eigenvalue[0])
            
    
    def set_h_emb(
        self,
        Lambda_c:MFType,
        D: MFType,
        h0_loc_matrix: MFType) -> None:
        """
        Set up the embedding Hamiltonian.

        Parameters
        ----------
        Lambda_c : dict[numpy.ndarray]
            Bath coupling. Each key in dictionary must follow
            :attr:`gf_struct`.
        D : dict[numpy.ndarray]
            Hybridization coupling. Each key in dictionary must follow
            :attr:`gf_struct`.
        h0_loc_matrix : dict[numpy.ndarray], optional
            Single-particle quadratic couplings of the c-electrons. Each key
            in dictionary must follow :attr:`gf_struct`.

        """
        if h0_loc_matrix is not None:
            self.set_h0_loc(h0_loc_matrix)
        self.set_h_bath(Lambda_c)
        self.set_h_hybr(D)
        self.set_h_int(self.V_int)
        
        self.h_emb = self.h0_loc + self.h_hybr + self.h_int + self.h_bath
        #self.h_emb = self.h0_loc + self.h_int #+ self.h_bath #+ self.h_hybr
        #self.h_emb = self.h0_loc
    
    def solve(self, test: bool = False) -> None:
        """
        Solve for the groundstate in the half-filled number sector of the embedding problem.
        if test==True, test the time cost of the solving process
        """
        # Check hermicities of the Hamiltonian
        for name, H in [
            ("h0_loc", self.h0_loc),
            ("h_bath", self.h_bath),
            ("h_hybr", self.h_hybr),
            ("h_int", self.h_int),
            ("h_emb", self.h_emb),
        ]:
            if not self.check_hermiticity(H):
                print(f"Warning: {name} is not Hermitian!")


        print("Start solving...") if test else None
        start_time = time.time() if test else None
        eigenvalue, eigenvector = eigsh(self.h_emb, k=1, which='SA')  # SA: smallest algebraic
        
        end_time = time.time() if test else None
        print("Solve finished, cost {:.2f}s".format(end_time - start_time)) if test else None

        self.gs_energy = eigenvalue[0]
        print("Ground state energy: {:.8f}".format(self.gs_energy)) if test else None
        self.gs_vector = eigenvector[:,0]
    
    def get_rho_f(self, bl: str) -> np.ndarray:
        """
        Return f-electron density matrix.
        delta_ab = <f_b f_a^dag> = <psi|f_b f_a^dag|psi>
        Parameters
        ----------
        bl : str
            Which block in :attr:`gf_struct` to return.
        
        Returns
        -------
        numpy.ndarray
            The f-electron density matrix :attr:`rho_f` from impurity.
        """
        L = self.L
        bl_size = self.gf_struct_dict[bl]
        self.rho_f[bl] = np.zeros([bl_size,bl_size],dtype=complex)
        # construct operator matrix for f-electrons
        idx0 = self.map_dict[bl]
        for a, b in product(range(bl_size), repeat=2):
            if bl.startswith("up"):
                i = L + 2 * (a + idx0)
                j = L + 2 * (b + idx0)
            elif bl.startswith("dn"):
                i = L + 2 * (a + idx0) + 1
                j = L + 2 * (b + idx0) + 1
            else:
                raise ValueError(f"Unknown block label {bl}")

            op_mat = self.create_c_c_dag_matrix(j, i) 
            val = self.gs_vector.conj().T @ (op_mat @ self.gs_vector) # f_b f_a^dag
            self.rho_f[bl][a, b] = val
        return self.rho_f[bl]

    def get_rho_c(self, bl: str) -> np.ndarray:
        """
        Return c-electron density matrix.
        <c_a^dag c_b> = <psi|c_a^dag c_b|psi>
        Parameters
        ----------
        bl : str
            Which block in :attr:`gf_struct` to return.
        
        Returns
        -------
        numpy.ndarray
            The c-electron density matrix :attr:`rho_c` from impurity.
        """
        L = self.L
        bl_size = self.gf_struct_dict[bl]
        self.rho_c[bl] = np.zeros([bl_size,bl_size],dtype=complex)
        # construct operator matrix for f-electrons
        idx0 = self.map_dict[bl]
        for a, b in product(range(bl_size), repeat=2):
            if bl.startswith("up"):
                i = 2 * (a + idx0)
                j = 2 * (b + idx0)
            elif bl.startswith("dn"):
                i = 2 * (a + idx0) + 1
                j = 2 * (b + idx0) + 1
            else:
                raise ValueError(f"Unknown block label {bl}")

            op_mat = self.create_c_dag_c_matrix(i, j) 
            val = self.gs_vector.conj().T @ (op_mat @ self.gs_vector) #
            self.rho_c[bl][a, b] = val
        return self.rho_c[bl]
    
    def get_rho_cf(self, bl: str) -> np.ndarray:
        """
        Return c-electron density matrix.
        <c_alpha^dag f_a> = <psi|c_alpha^dag f_a|psi>
        Parameters
        ----------
        bl : str
            Which block in :attr:`gf_struct` to return.
        
        Returns
        -------
        numpy.ndarray
            The c,f-electron density matrix :attr:`rho_cf` from impurity.
        """
        L = self.L
        bl_size = self.gf_struct_dict[bl]
        self.rho_cf[bl] = np.zeros([bl_size,bl_size],dtype=complex)
        # construct operator matrix for f-electrons
        idx0 = self.map_dict[bl]
        for alpha, a in product(range(bl_size), repeat=2):
            if bl.startswith("up"):
                i = 2 * (alpha  + idx0)
                j = L + 2 * (a + idx0)
            elif bl.startswith("dn"):
                i = 2 * (alpha + idx0) + 1
                j = L + 2 * (a + idx0) + 1
            else:
                raise ValueError(f"Unknown block label {bl}")

            op_mat = self.create_c_dag_c_matrix(i, j) 
            val = self.gs_vector.conj().T @ (op_mat @ self.gs_vector)
            self.rho_cf[bl][alpha, a] = val
        return self.rho_cf[bl]
    
    def get_Seff(self, bl: str):
        # compute the effective spin for a given block label, for example: "up_T"

        def Pauli_mat(): # Pauli matrices for spin-1/2 system
            s0 = np.eye(2)
            sx = np.array([[0,1],[1,0]],dtype=complex)
            sy = np.array([[0,-1j],[1j,0]],dtype=complex)
            sz = np.array([[1,0],[0,-1]],dtype=complex)
            return s0,sx,sy,sz
        # Construction of the S^2 operator: (Sx^2 + Sy^2 + Sz^2, S = \sum_a S_a for multiorbital)
        row, col, data = [], [], []
        bl_size = self.gf_struct_dict[bl] # number of orbitals in this block
        idx0 = self.map_dict[bl] # index of the first orbital in this block
        s0,sx,sy,sz = Pauli_mat() # get the Pauli matrices
        for istate, state in enumerate(self.basis): # loop over all states
            for ia, ib in product(range(bl_size),repeat=2): # loop over all pairs of orbitals
                idxa = idx0 + ia
                idxb = idx0 + ib
                for s, s1, t, t1 in product(range(2),repeat=4):
                    # apply the Pauli matrices to the state vector
                    idx1 = 2*idxa + s
                    idx2 = 2*idxa + s1
                    idx3 = 2*idxb + t
                    idx4 = 2*idxb + t1
                    new_state, sign = self.apply_c_dag_c_c_dag_c(state,idx1,idx2,idx3,idx4)
                    amplitude = 1/4 * (sx[s,s1]*sx[t,t1]+sy[s,s1]*sy[t,t1]+sz[s,s1]*sz[t,t1])
                    if new_state in self.index:
                        jstate = self.index[new_state]
                        row.append(jstate)
                        col.append(istate)
                        data.append(sign * amplitude)
        S_sq = self.build_sparse_matrix(row,col,data) # operator in matrix form
        S_sq_val = self.gs_vector.conj().T @ S_sq @ self.gs_vector
        # Solve for S^{\hat}^2 = S(S+1)
        S = (np.sqrt(4*S_sq_val+1)-1)/2
        return S
        



class EmbeddingSparseDummy:
    """
    Dummy embedding solver referencing an existing SolveEmbeddingSparse object.

    This dummy does not solve the embedding problem but retrieves data
    (e.g., density matrices) from a reference embedding, possibly applying rotations.

    Parameters
    ----------
    embedding : SolveEmbeddingSparse
        The actual embedding solver to reference.
    rotations : list[callable], optional
        Rotation functions to apply to density matrices (rho_c, rho_f, rho_cf).
    """

    def __init__(self, embedding, rotations=None):
        if rotations is None:
            rotations = []
        self.embedding = embedding
        self.rotations = rotations

    def set_h_emb(self, *args, **kwargs):
        pass  # dummy does not build the Hamiltonian

    def solve(self, *args, **kwargs):
        pass  # dummy does not solve

    def get_rho_f(self, bl: str) -> np.ndarray:
        rho = self.embedding.get_rho_f(bl)
        for rot in self.rotations:
            rho = rot(rho)
        return rho

    def get_rho_c(self, bl: str) -> np.ndarray:
        rho = self.embedding.get_rho_c(bl)
        for rot in self.rotations:
            rho = rot(rho)
        return rho

    def get_rho_cf(self, bl: str) -> np.ndarray:
        rho = self.embedding.get_rho_cf(bl)
        for rot in self.rotations:
            rho = rot(rho)
        return rho

