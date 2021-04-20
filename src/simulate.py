import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
import coords
import fourier
import transfer
import gauss_forward_model


def simulate_slice(
    map_r,
    psize,
    n_particles,
    N_crop=None,
    snr=0.1,
    do_snr=True,
    do_ctf=True,
    df_min=15000,
    df_max=20000,
    df_diff_min=100,
    df_diff_max=500,
    df_ang_min=0,
    df_ang_max=360,
    kv=300,
    cs=2.0,
    ac=0.1,
    phase=0,
    bf=0,
    do_log=True,
    random_seed=0,
):
    """
    Simulates a particle stack from a 3D voxelized array using the Fourier slice theorem (projection assumption).

    Parameters
    ----------
        map_r : numpy.ndarray, shape (N,N,N)
            3D real space map of shape 
        psize : float
            pixel size in Angstroms
        n_particles: int
            number of particles to simulate
        N_crop : in
            number of pixels to crop to
        snr : float
            signal to noise ratio. defined as the std(signal) / std(noise). the signal is the projections (with ctf if that's included)
        do_snr : bool 
            add nosie when True
        do_ctf : bool 
            add ctf when True
        df_min : float 
            defocus range min
        df_max : float 
            defocus range max
        df_diff_min : float 
            defocus difference range min
        df_diff_max : float 
            defocus difference range max
        df_ang_min : float 
            angle of astigmatism range min
        df_ang_max : float 
            angle of astigmatism range max
        kv : float 
            voltate of microscope in kilo electron volts (typicaly 120, 200, 300 on electron microscopes)
        cs : float 
            spherical aberation constant in mm
        ac : float 
            amplitude contrast
        phase : float 
            ctf phase
        bf : float 
            B-factor for CTF envelope. no envelope when 0
        do_log : bool 
            log progress if True. TODO: add to logger instead of print
        random_seed : int 
            random_seed for np.random.seed(random_seed)

    Returns
    ------- 
        projs_r: numpy.ndarray, shape (n_particles,N_crop,N_crop)
            particle stack of simulated projections
        projs_r_noise : numpy.ndarray, shape (n_particles,N_crop,N_crop)
            particle stack of simulated projections with noise
        meta_data_df : pandas.DataFrame
            metadata of simulated particles, including sampled ctf
    """
    assert len(map_r.shape) == 3
    assert np.unique(map_r.shape).size == 1
    N = map_r.shape[0]
    assert N % 2 == 0, "even pixel length"

    map_f = fourier.do_fft(map_r)

    N = map_f.shape[0]
    xyz = coords.coords_n_by_d(np.arange(-N // 2, N // 2), d=3)
    idx_z0 = xyz[:, -1] == 0
    xy0 = xyz[idx_z0]

    if random_seed is not None:
        np.random.seed(random_seed)
    rots, qs = coords.uniform_rotations(n_particles)

    projs_f = np.zeros((n_particles, N, N), dtype=np.complex64)
    for idx in range(n_particles):
        if do_log and idx % max(1, (n_particles // 10)) == 0:
            print(idx)
        rot = rots[idx, :, :]
        xy0_rot = rot.dot(xy0.T).T
        projs_f[idx] = (
            map_coordinates(map_f.real, xy0_rot.T + N // 2, order=1).astype(
                np.complex64
            )
            + 1j
            * map_coordinates(map_f.imag, xy0_rot.T + N // 2, order=1).astype(
                np.complex64
            )
        ).reshape(
            N, N
        )  # important to keep order=1 for speed. linear is good enough

    if do_ctf:
        ctfs, df1s, df2s, df_ang_deg = transfer.random_ctfs(
            N=N,
            psize=psize,
            n_particles=n_particles,
            df_min=df_min,
            df_max=df_max,
            df_diff_min=df_diff_min,
            df_diff_max=df_diff_max,
            df_ang_min=df_ang_min,
            df_ang_max=df_ang_max,
            kv=kv,
            cs=cs,
            ac=ac,
            phase=phase,
            bf=bf,
            do_log=do_log,
        )

        projs_f *= ctfs

    if N_crop is not None:
        i, f = N // 2 - N_crop // 2, N // 2 + N_crop // 2
        projs_r = fourier.do_ifft(projs_f[:, i:f, i:f], d=2, batch=True)
        psize = psize * N / N_crop
    else:
        projs_r = fourier.do_ifft(projs_f, d=2, batch=True)

    if do_snr:
        signal = np.std(projs_r)
        noise = signal / snr
        projs_r_noise = np.random.normal(loc=projs_r, scale=noise)
    else:
        projs_r_noise = projs_r

    meta_data_d = {
        "N": N_crop,
        "psize": psize_crop,
        "snr": snr,
        "rotation_quaternion": [np.array2string(q) for q in qs],
    }
    if do_ctf:
        meta_data_d.update(
            {
                "df1_A": df1s,
                "df2_A": df2s,
                "df_ang_deg": df_ang_deg,
                "kev": kv,
                "ac": ac,
                "cs_mm": cs,
            }
        )
    meta_data_df = pd.DataFrame(meta_data_d)

    return (projs_r, projs_r_noise, meta_data_df)


def simulate_atoms(
    atoms,
    N,
    psize,
    n_particles,
    sigma=1,  # in pixels
    n_trunc=None,  # in pixels
    snr=0.1,
    do_snr=True,
    do_ctf=True,
    df_min=15000,
    df_max=20000,
    df_diff_min=100,
    df_diff_max=500,
    df_ang_min=0,
    df_ang_max=360,
    kv=300,
    cs=2.0,
    ac=0.1,
    phase=0,
    bf=0,
    do_log=True,
    random_seed=0,
):
    """
    Simulates a particle stack from 3D atomic coordinates (superposition approximiation).

    Parameters
    ----------
        atoms : numpy.ndarray
            xyz coordinates of atoms of shape (3,n_atoms)
        N : int
            number of pixels for projections of shape (N,N). #TODO add N_crop (after ctf)
        psize : float
            pixel size in Angstroms
        n_particles : int
            number of particles to simulate
        sigma : float
            spread of Gaussian blob of atom. atomic B-factor. in units of pixels (not angstroms) 
            # TODO: extend to have per atom sigmas (and n_trunc)
        n_trunc : int
            size of patch around Gaussian blob, outside of which the Gaussian is truncated to zero. Reasonable value of int(6*sigma), because gaussian near zero 3 sigma away from mean 
        snr : float
            signal to noise ratio. defined as the std(signal) / std(noise). the signal is the projections (with ctf if that's included)
        do_snr : bool 
            add nosie when True
        do_ctf : bool 
            add ctf when True
        df_min : float 
            defocus range min
        df_max : float 
            defocus range max
        df_diff_min : float 
            defocus difference range min
        df_diff_max : float 
            defocus difference range max
        df_ang_min : float 
            angle of astigmatism range min
        df_ang_max : float 
            angle of astigmatism range max
        kv : float 
            voltate of microscope in kilo electron volts (typicaly 120, 200, 300 on electron microscopes)
        cs : float 
            spherical aberation constant in mm
        ac : float 
            amplitude contrast
        phase : float 
            ctf phase
        bf : float 
            B-factor for CTF envelope. no envelope when 0
        do_log : bool 
            log progress if True. TODO: add to logger instead of print
        random_seed : int 
            random_seed for np.random.seed(random_seed)

    Returns
    ------- 
        projs_r: numpy.ndarray, shape (n_particles,N_crop,N_crop)
            particle stack of simulated projections
        projs_r_noise : numpy.ndarray, shape (n_particles,N_crop,N_crop)
            particle stack of simulated projections with noise
        meta_data_df : pandas.DataFrame
            metadata of simulated particles, including sampled ctf
    """
    assert atoms.shape[0] == 3
    n_atoms = atoms.shape[-1]
    assert N % 2 == 0, "even pixel length"

    xy = coords.coords_n_by_d(np.arange(-N // 2, N // 2), d=2) # in pixel units, not angstroms
    if n_trunc is None:
        n_trunc = round(6 * sigma)

    if random_seed is not None:
        np.random.seed(random_seed)
    rots, qs = coords.uniform_rotations(n_particles)
    g_2d_gpu = gauss_forward_model.make_proj_gpu(
        atoms / psize, xy, N, n_particles, sigma, n_trunc, rots
    )  # TODO verify general for psize (work in pixel units by converting atoms to pixel units)

    if do_log:
        print("copy_to_host")
    projs_r = g_2d_gpu.copy_to_host().reshape(n_particles, N, N)

    # CTF
    # to avoid CTF aliasing, ensure N is large
    if do_ctf:
        if do_log:
            print("CTF")
        projs_f = fourier.do_fft(
            projs_r, d=2, batch=True
        )  # TODO: may need to zero pad here for CTF if small N (e.g. 128 or less)
        ctfs, df1s, df2s, df_ang_deg = transfer.random_ctfs(
            N=N,
            psize=psize,
            n_particles=n_particles,
            df_min=df_min,
            df_max=df_max,
            df_diff_min=df_diff_min,
            df_diff_max=df_diff_max,
            df_ang_min=df_ang_min,
            df_ang_max=df_ang_max,
            kv=kv,
            cs=cs,
            ac=ac,
            phase=phase,
            bf=bf,
            do_log=do_log,
        )

        projs_f *= ctfs
        projs_r = fourier.do_ifft(projs_f, d=2, batch=True)

    if do_snr:
        signal = np.std(projs_r)
        noise = signal / snr
        projs_r_noise = np.random.normal(loc=projs_r, scale=noise)
    else:
        projs_r_noise = projs_r

    meta_data_d = {
        "N": N,
        "psize": psize,
        "snr": snr,
        "rotation_quaternion": [np.array2string(q) for q in qs],
    }
    if do_ctf:
        meta_data_d.update(
            {
                "df1_A": df1s,
                "df2_A": df2s,
                "df_ang_deg": df_ang_deg,
                "kev": kv,
                "ac": ac,
                "cs_mm": cs,
            }
        )
    meta_data_df = pd.DataFrame(meta_data_d)

    return (projs_r, projs_r_noise, meta_data_df)
