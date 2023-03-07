import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

############################################################################################################
# |\nabla u| = f

# ((u - a1)^+)^2 + ((u - a2)^+)^2 = f^2 h^2


def calculate_unique_solution(a, b, f, h):

    d = abs(a - b)
    if d >= f * h:
        return min(a, b) + f * h
    else:
        return (a + b + np.sqrt(2 * f * f * h * h - (a - b) ** 2)) / 2


def sweeping_over_I_J_K(u, I, J, f, h):
    # print("Sweeping start...")
    m = len(I)
    n = len(J)
    for i in I:
        for j in J:
            if i == 0:
                uxmin = u[i + 1, j]
            elif i == m - 1:
                uxmin = u[i - 1, j]
            else:
                uxmin = np.min([u[i - 1, j], u[i + 1, j]])

            if j == 0:
                uymin = u[i, j + 1]
            elif j == n - 1:
                uymin = u[i, j - 1]
            else:
                uymin = np.min([u[i, j - 1], u[i, j + 1]])

            u_new = calculate_unique_solution(uxmin, uymin, f[i, j], h)

            u[i, j] = np.min([u_new, u[i, j]])

    return u


def sweeping(u, f, h):

    s = 1.0 / f  ## slowness

    m, n = u.shape
    I = list(range(m))
    iI = I[::-1]
    J = list(range(n))
    iJ = J[::-1]

    u = sweeping_over_I_J_K(u, I, J, s, h)
    u = sweeping_over_I_J_K(u, iI, J, s, h)
    u = sweeping_over_I_J_K(u, iI, iJ, s, h)
    u = sweeping_over_I_J_K(u, I, iJ, s, h)

    return u


def eikonal_solve(u, f, h):

    for i in range(50):
        u_old = np.copy(u)
        u = sweeping(u, f, h)

        err = np.max(np.abs(u - u_old))

        if err < 1e-6:
            break

    return u


def normalize(vars_, bounds):
    mean = torch.tensor(
        [
            [
                (bounds[0][0] + bounds[0][1]) / 2,
                (bounds[1][0] + bounds[1][1]) / 2,
                (bounds[2][0] + bounds[2][1]) / 2,
            ],
        ],
        dtype=torch.float32,
    )
    std = torch.tensor(
        [
            [
                (bounds[0][1] - bounds[0][0]) / 2,
                (bounds[1][1] - bounds[1][0]) / 2,
                (bounds[2][1] - bounds[2][0]) / 2,
            ]
        ],
        dtype=torch.float32,
    )
    vars = (vars_ - mean) / std
    vars = torch.tanh(vars)
    vars = (vars * std) + mean

    return vars


# def traveltime(vars_, station_locs, phase_type, up, us, h, rgrid, zgrid, bounds=None):

#     def interp(X, h, input):
#         # the origin is (0,0)
#         ir0 = torch.floor(input[0].div(h)).type(torch.long)
#         ir1 = ir0 + 1
#         iz0 = torch.floor(input[1].div(h)).type(torch.long)
#         iz1 = iz0 + 1
#         if iz0 >= zgrid.shape[1]:
#             iz0 = zgrid.shape[1] - 1
#         if iz1 >= zgrid.shape[1]:
#             iz1 = zgrid.shape[1] - 1
#         r0 = ir0 * h
#         z0 = iz0 * h

#         Ia = X[ir0, iz1]
#         Ib = X[ir1, iz1]
#         Ic = X[ir0, iz0]
#         Id = X[ir1, iz0]

#         return ((Ib-Ia) * (input[0]-r0)/h + Ia - (Id-Ic) * (input[0]-r0)/h - Ic) * (input[1]-z0)/h + (Id-Ic) * (input[0]-r0)/h + Ic

#     if bounds is not None:
#         vars = normalize(vars_, bounds)
#     else:
#         vars = vars_

#     r = torch.sqrt(torch.sum((vars[0, :2] - station_locs[:, :2]) ** 2, dim=-1)).unsqueeze(1)
#     z = torch.abs(vars[0, 2] - station_locs[:, 2]).unsqueeze(1)
#     t = torch.cat([interp(up, h, torch.cat([r[i], z[i]])).unsqueeze(0) if phase_type[i]=='p' else interp(us, h, torch.cat([r[i], z[i]])).unsqueeze(0) for i, _ in enumerate(phase_type)], 0)
#     return t


# def invert_location(
#     data, event_t0, event_locs, station_locs, phase_type, weight, up, us, h, rgrid, zgrid, bounds=None
# ):
#     t0_ = torch.tensor(event_t0, dtype=torch.float32, requires_grad=True)
#     loc_ = torch.tensor(event_locs, dtype=torch.float32, requires_grad=True)
#     if bounds is not None:
#         loc = normalize(loc_, bounds)
#         t0 = t0_
#     else:
#         loc = loc_
#         t0 = t0_
#     station_locs = torch.tensor(station_locs, dtype=torch.float32)
#     weight = torch.tensor(weight, dtype=torch.float32).squeeze(1)
#     data = torch.tensor(data, dtype=torch.float32)
#     rgrid = torch.tensor(rgrid, dtype=torch.float32)
#     zgrid = torch.tensor(zgrid, dtype=torch.float32)
#     up = torch.tensor(up, dtype=torch.float32)
#     us = torch.tensor(us, dtype=torch.float32)
#     optimizer = optim.LBFGS(params=[t0_, loc_], max_iter=100, line_search_fn="strong_wolfe")

#     def closure():
#         optimizer.zero_grad()
#         tt = t0_[0] + traveltime(loc_, station_locs, phase_type, up, us, h, rgrid, zgrid, bounds=bounds)
#         loss = F.huber_loss(data, tt, reduction='none') * weight
#         loss = loss.sum() / (weight.sum() + torch.tensor(1e-6))
#         loss.backward(retain_graph=True)
#         return loss

#     optimizer.step(closure)

#     if bounds is not None:
#         loc = normalize(loc_, bounds)
#         t0 = t0_
#     else:
#         loc = loc_
#         t0 = t0_
        
#     return torch.cat((loc.squeeze(0), t0), 0).detach().numpy()


def _interp(time_table, r, z, rgrid, zgrid, h):
    ir0 = (r - rgrid[0, 0]).div(h, rounding_mode='floor').clamp(0, rgrid.shape[0] - 2).long()
    iz0 = (z - zgrid[0, 0]).div(h, rounding_mode='floor').clamp(0, zgrid.shape[1] - 2).long()
    ir1 = ir0 + 1
    iz1 = iz0 + 1

    ## https://en.wikipedia.org/wiki/Bilinear_interpolation
    x1 = ir0 * h + rgrid[0, 0]
    x2 = ir1 * h + rgrid[0, 0]
    y1 = iz0 * h + zgrid[0, 0]
    y2 = iz1 * h + zgrid[0, 0]

    Q11 = time_table[ir0, iz0]
    Q12 = time_table[ir0, iz1]
    Q21 = time_table[ir1, iz0]
    Q22 = time_table[ir1, iz1]

    t = (
        1
        / (x2 - x1)
        / (y2 - y1)
        * (
            Q11 * (x2 - r) * (y2 - z)
            + Q21 * (r - x1) * (y2 - z)
            + Q12 * (x2 - r) * (z - y1)
            + Q22 * (r - x1) * (z - y1)
        )
    )

    return t


def traveltime(event_loc, station_loc, time_table, rgrid, zgrid, h, **kwargs):
    r = torch.sqrt(torch.sum((event_loc[:, :2] - station_loc[:, :2]) ** 2, dim=-1, keepdims=True))
    z = event_loc[:, 2:] - station_loc[:, 2:]
    if (event_loc[:, 2:] < 0).any():
        print(f"Warning: depth is defined as positive down: {event_loc[:, 2:].detach().numpy()}")

    tt = _interp(time_table, r, z, rgrid, zgrid, h)

    return tt


def invert_location(
    phase_time,
    event_loc0,
    station_loc,
    phase_type,
    weight,
    up,
    us,
    h,
    rgrid,
    zgrid,
    bounds=None,
    device="cpu",
    add_eqt=False,
    gamma=0.1,
    max_iter=100,
    convergence=1e-6,
):
    event_loc = torch.tensor(event_loc0, dtype=torch.float32, requires_grad=True, device=device)
    if bounds is not None:
        bounds = torch.tensor(bounds, dtype=torch.float32, device=device)

    rgrid = torch.tensor(rgrid, dtype=torch.float32)
    zgrid = torch.tensor(zgrid, dtype=torch.float32)
    up = torch.tensor(up, dtype=torch.float32)
    us = torch.tensor(us, dtype=torch.float32)
    p_index = torch.arange(len(phase_type), device=device)[phase_type == "p"]
    s_index = torch.arange(len(phase_type), device=device)[phase_type == "s"]
    time = torch.tensor(phase_time, dtype=torch.float32, device=device)
    loc = torch.tensor(station_loc, dtype=torch.float32, device=device)
    weight = torch.tensor(weight, dtype=torch.float32, device=device)
    obs_p = time[p_index]
    obs_s = time[s_index]
    loc_p = loc[p_index]
    loc_s = loc[s_index]
    weight_p = weight[p_index]
    weight_s = weight[s_index]

    # %% optimization
    optimizer = torch.optim.LBFGS(params=[event_loc], max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_change=convergence)

    def closure():
        optimizer.zero_grad()
        if bounds is not None:
            loc0_ = torch.max(torch.min(event_loc[:, :-1], bounds[:, 1]), bounds[:, 0])
        else:
            loc0_ = event_loc[:, :-1]
        loc0_ = torch.nan_to_num(loc0_, nan=0)
        t0_ = event_loc[:, -1:]
        if len(p_index) > 0:
            tt_p = traveltime(loc0_, loc_p, up, rgrid, zgrid, h, sigma=1)
            pred_p = t0_ + tt_p
            loss_p = torch.mean(F.huber_loss(obs_p, pred_p, reduction="none") * weight_p)
            if add_eqt:
                dd_tt_p = tt_p.unsqueeze(-1) - tt_p.unsqueeze(-2)
                dd_time_p = obs_p.unsqueeze(-1) - obs_p.unsqueeze(-2)
                loss_p += gamma * torch.mean(
                    F.huber_loss(dd_tt_p, dd_time_p, reduction="none") * weight_p.unsqueeze(-1) * weight_p.unsqueeze(-2)
                )
            # loss_p = F.mse_loss(time_p, tt_p)
        else:
            loss_p = 0
        if len(s_index) > 0:
            tt_s = traveltime(loc0_, loc_s, us, rgrid, zgrid, h, sigma=1)
            pred_s = t0_ + tt_s
            loss_s = torch.mean(F.huber_loss(obs_s, pred_s, reduction="none") * weight_s)
            if add_eqt:
                dd_tt_s = tt_s.unsqueeze(-1) - tt_s.unsqueeze(-2)
                dd_time_s = obs_s.unsqueeze(-1) - obs_s.unsqueeze(-2)
                loss_s += gamma * torch.mean(
                    F.huber_loss(dd_tt_s, dd_time_s, reduction="none") * weight_s.unsqueeze(-1) * weight_s.unsqueeze(-2)
                )
            # loss_s = F.mse_loss(time_s, tt_s)
        else:
            loss_s = 0
        loss = loss_p + loss_s
        loss.backward()
        return loss

    optimizer.step(closure)
    loss = closure().item()

    event_loc = event_loc.detach().cpu()
    if bounds is not None:
        event_loc[:, :-1] = torch.max(torch.min(event_loc[:, :-1], bounds[:, 1]), bounds[:, 0])

    return event_loc.detach().numpy()