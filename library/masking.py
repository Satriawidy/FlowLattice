from .initial import *


def make_checker_mask(shape, parity):                       #Simple checker mask (for scalar phi4)
    checker = torch.ones(shape, dtype=torch.uint8) - parity #Toy case, parity = 0, checker = 1
    checker[::2, ::2] = parity                              #Now quarter of config become parity = 0
    checker[1::2, 1::2] = parity                            #Now half (checkered) of config become 0
    return checker.to(torch_device)

def make_2d_link_active_stripes(shape, mu, off):
    assert len(shape) == 2+1, 'need to pass shape suitable for 2D gauge theory'
    assert shape[0] == len(shape[1:]), 'first dim of shape must be Nd'
    assert mu in (0, 1), 'must be 0 or 1'

    mask = np.zeros(shape).astype(np.uint8)
    if mu == 0:
        mask[mu, :, 0::4] = 1
    elif mu == 1:
        mask[mu, 0::4] = 1
    nu = 1 - mu
    mask = np.roll(mask, off, axis = nu + 1)
    return torch.from_numpy(mask.astype(float_dtype)).to(torch_device)

def make_single_stripes(shape, mu, off):
    """
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
    
    where vertical is thee 'mu' direction. Vector of 1 is repeated every 4.
    The pattern is offset in perpendicular to the mu direction by 'off' (mod 4).
    """
    assert len(shape) == 2, 'need to pass 2D shape'
    assert mu in (0,1), 'mu must be 0 or 1'

    mask = np.zeros(shape).astype(np.uint8)
    if mu == 0:
        mask[:,0::4] = 1
    elif mu == 1:
        mask[0::4] = 1
    mask = np.roll(mask, off, axis = 1 - mu)
    return torch.from_numpy(mask).to(torch_device)

def make_double_stripes(shape, mu, off):
    """
    Double stripes mask looks like
      1 1 0 0 1 1 0 0 
      1 1 0 0 1 1 0 0 
      1 1 0 0 1 1 0 0 
      1 1 0 0 1 1 0 0 
    
    where vertical is thee 'mu' direction. The pattern is offset in perpendicular 
    to the mu direction by 'off' (mod 4).
    """
    assert len(shape) == 2, 'need to pass 2D shape'
    assert mu in (0,1), 'mu must be 0 or 1'

    mask = np.zeros(shape).astype(np.uint8)
    if mu == 0:
        mask[:,0::4] = 1
        mask[:,1::4] = 1
    elif mu == 1:
        mask[0::4] = 1
        mask[1::4] = 1
    mask = np.roll(mask, off, axis = 1 - mu)
    return torch.from_numpy(mask).to(torch_device)

def make_plaq_masks(mask_shape, mask_mu, mask_off):
    mask = {}
    mask['frozen'] = make_double_stripes(mask_shape, mask_mu, mask_off+1)
    mask['active'] = make_single_stripes(mask_shape, mask_mu, mask_off)
    mask['passive'] = 1 - mask['frozen'] - mask['active']
    return mask