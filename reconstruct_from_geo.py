import numpy as np
import matplotlib.pyplot as plt

from torax._src import constants
from torax._src.geometry import geometry
from torax._src.geometry import geometry_loader
from torax._src.geometry import standard_geometry


def _construct_intermediates_from_chease(
        geometry_directory: str | None,
        geometry_file: str,
        Ip_from_parameters: bool,
        n_rho: int,
        R_major: float,
        a_minor: float,
        B_0: float,
        hires_factor: int,
) -> standard_geometry.StandardGeometryIntermediates:
    chease_data = geometry_loader.load_geo_data(
        geometry_directory, geometry_file, geometry_loader.GeometrySource.CHEASE
    )

    psiunnormfactor = R_major ** 2 * B_0 * 2 * np.pi
    psi = chease_data['PSIchease=psi/2pi'] * psiunnormfactor
    Ip_chease = (
            chease_data['Ipprofile'] / constants.CONSTANTS.mu_0 * R_major * B_0
    )

    Phi = (chease_data['RHO_TOR=sqrt(Phi/pi/B0)'] * R_major) ** 2 * B_0 * np.pi

    R_in_chease = chease_data['R_INBOARD'] * R_major
    R_out_chease = chease_data['R_OUTBOARD'] * R_major
    # toroidal field flux function
    F = chease_data['T=RBphi'] * R_major * B_0

    int_dl_over_Bp = (
            chease_data['Int(Rdlp/|grad(psi)|)=Int(Jdchi)'] * R_major / B_0
    )
    flux_surf_avg_1_over_R = chease_data['<1/R>profile'] / R_major
    flux_surf_avg_1_over_R2 = chease_data['<1/R**2>'] / R_major ** 2
    # COCOS > 10: <|\nabla \psi|> = 2\pi<R B_p>
    flux_surf_avg_grad_psi2_over_R2 = (
            chease_data['<Bp**2>'] * B_0 ** 2 * (4 * np.pi ** 2)
    )
    flux_surf_avg_grad_psi = (
            chease_data['<|grad(psi)|>'] * psiunnormfactor / R_major
    )
    flux_surf_avg_grad_psi2 = (
            chease_data['<|grad(psi)|**2>'] * psiunnormfactor ** 2 / R_major ** 2
    )
    flux_surf_avg_B2 = chease_data['<B**2>'] * B_0 ** 2
    flux_surf_avg_1_over_B2 = chease_data['<1/B**2>'] / B_0 ** 2

    rhon = np.sqrt(Phi / Phi[-1])
    vpr = 4 * np.pi * Phi[-1] * rhon / (F * flux_surf_avg_1_over_R2)

    return standard_geometry.StandardGeometryIntermediates(
        geometry_type=geometry.GeometryType.CHEASE,
        Ip_from_parameters=Ip_from_parameters,
        R_major=np.array(R_major),
        a_minor=np.array(a_minor),
        B_0=np.array(B_0),
        psi=psi,
        Ip_profile=Ip_chease,
        Phi=Phi,
        R_in=R_in_chease,
        R_out=R_out_chease,
        F=F,
        int_dl_over_Bp=int_dl_over_Bp,
        flux_surf_avg_1_over_R=flux_surf_avg_1_over_R,
        flux_surf_avg_1_over_R2=flux_surf_avg_1_over_R2,
        flux_surf_avg_grad_psi2_over_R2=flux_surf_avg_grad_psi2_over_R2,
        flux_surf_avg_grad_psi=flux_surf_avg_grad_psi,
        flux_surf_avg_grad_psi2=flux_surf_avg_grad_psi2,
        flux_surf_avg_B2=flux_surf_avg_B2,
        flux_surf_avg_1_over_B2=flux_surf_avg_1_over_B2,
        delta_upper_face=chease_data['delta_upper'],
        delta_lower_face=chease_data['delta_bottom'],
        elongation=chease_data['elongation'],
        vpr=vpr,
        n_rho=n_rho,
        hires_factor=hires_factor,
        diverted=None,
        connection_length_target=None,
        connection_length_divertor=None,
        angle_of_incidence_target=None,
        R_OMP=None,
        R_target=None,
        B_pol_OMP=None,
        z_magnetic_axis=None,
    )


def plot_geometry_layers(geo_intermediates, title="Torax Geometry Layers (Miller Reconstruction)"):
    theta = np.linspace(0, 2 * np.pi, 360)

    Phi = geo_intermediates.Phi
    Phi_safe = np.where(Phi == 0, 1e-9, Phi)
    rhon = np.sqrt(Phi_safe / Phi[-1])
    rhon[0] = 0.0

    a_tot = geo_intermediates.a_minor.item() if geo_intermediates.a_minor.ndim == 0 else geo_intermediates.a_minor[0]

    kappa_profile = geo_intermediates.elongation
    delta_avg_profile = (geo_intermediates.delta_upper_face + geo_intermediates.delta_lower_face) / 2.0
    R_geo_profile = (geo_intermediates.R_in + geo_intermediates.R_out) / 2.0

    n_rho_points = len(rhon)

    fig, ax = plt.subplots(figsize=(7, 9))

    indices_to_plot = np.unique(np.linspace(1, n_rho_points - 1, num=10, dtype=int))

    print(f"Plotting geometry for R_major={geo_intermediates.R_major}, a_minor={a_tot}")

    for i, idx in enumerate(indices_to_plot):
        rho_local = rhon[idx]
        r_local = a_tot * rho_local
        k_local = kappa_profile[idx]
        d_local = delta_avg_profile[idx]
        R_c_local = R_geo_profile[idx]
        Z_c_local = 0.0

        # R(theta) = R_geo + r * cos(theta + arcsin(delta) * sin(theta))
        # Z(theta) = Z_geo + kappa * r * sin(theta)

        d_local_safe = np.clip(d_local, -0.99, 0.99)

        R_surf = R_c_local + r_local * np.cos(theta + np.arcsin(d_local_safe) * np.sin(theta))
        Z_surf = Z_c_local + k_local * r_local * np.sin(theta)

        is_lcfs = (idx == indices_to_plot[-1])
        if is_lcfs:
            ax.plot(R_surf, Z_surf, 'r-', linewidth=2.5, label='LCFS (Boundary, $\\rho_N$=1.0)')
            ax.plot(np.max(R_surf), 0, 'ro', markersize=8)
        else:
            alpha_val = 0.3 + 0.4 * rho_local
            ax.plot(R_surf, Z_surf, 'b--', linewidth=1.0, alpha=alpha_val)

    ax.plot(R_geo_profile[0], 0, 'k+', markersize=12, markeredgewidth=2, label='Magnetic Axis (Approx)')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Major Radius R [m]', fontsize=12)
    ax.set_ylabel('Vertical Position Z [m]', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, which='both', linestyle=':', linewidth=0.8)
    ax.legend(loc='upper right', frameon=True, shadow=True)

    R_min_plot = np.min(geo_intermediates.R_in) * 0.9
    R_max_plot = np.max(geo_intermediates.R_out) * 1.05
    Z_max_plot = np.max(kappa_profile * a_tot) * 1.1
    ax.set_xlim(R_min_plot, R_max_plot)
    ax.set_ylim(-Z_max_plot, Z_max_plot)

    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    geo_intermediates = _construct_intermediates_from_chease(
        geometry_directory=None,
        geometry_file='iterhybrid.mat2cols',
        Ip_from_parameters=True,
        R_major=6.2,
        a_minor=2.0,
        B_0=5.3,
        n_rho=25,
        hires_factor=4
    )
    print("Successfully constructed StandardGeometryIntermediates.")

    plot_geometry_layers(geo_intermediates, title="ITER Hybrid Geometry (Torax Reconstruction)")

    plt.show()
