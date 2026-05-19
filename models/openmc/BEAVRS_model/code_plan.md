# The mathematical plan

The mathematical plan for the project is described below.

## A Constants

Neutron speed, $v$ is also considered constant for now, since we are not doing energy groups yet.

## B. Inputs

The reactor dimensions and material composition are taken as constant inputs.

## C. Start of the solver loop

At time $t$, we have temperature $T(t)$.

## D. Nuclear data acquisition

For a given material composition at temp $T(t)$, use OpenMC to generate the necessary nuclear data at reasonable temperature intervals from a nuclear data library like JEFF4.0. Then, pull in the data and interpolate/curve-fit along the temperature axis.

1. $\nu\Sigma_f(t)$,
2. $\Sigma_a(t)$,
3. $\Sigma_s(t)$,
4. $D(t)$,
5. $\beta_{i}(t)$,
6. $\lambda_{i}(t)$

## E. Nuclear properties calculations

Calculate

1. $\Lambda(t)$,
2. $\beta(t)$,
3. $k_{\text{eff}}(t)$,
4. $\rho(t)$,
5. $B^2(t)$

## F. Kinetics

Solve kinetics time step $\Delta t$, get power $P(t+\Delta t)$

## G. Thermalhydraulics

Solve TH time step $\Delta t$, get temp $T(t+\Delta t)$

## H. Time march

Now we are at time $t = t+\Delta t$

But for the initial implementation, we will skip step H, assuming some value of T in step C. We introduce a little bit of high fidelity by calculating most of the stuff in steps D and E.
