## Drying ink model
We investigate the the situation of ink drying on a smooth surface. 
Let $\phi(x,t)$ be an an order parameter describing if the surface is wet ($\phi = 1$) or dry ($\phi = -1$). Then the solvent can be modelled using an amended **Allen-Cahn equation**: 
<p align=center>
$\frac{\partial \phi}{\partial t} = M \left( \sigma \Delta \phi - \frac{\partial f(\phi,n,t)}{\partial \phi} + \mu \right), $
<p>
where f is the local free energy, M the evaporation rate, $\sigma$ a parameter related to the surface tension and $\mu$ describes the amount of moisture in the air. 
An usual choice for the free energy is $f = f_1 = -1/2 \phi^2 + 1/4 \phi^4$ which has minima for $\phi = \pm 1$.

A second field $n(x,t)$ describes the distribution of the ink particle on the surface and is modelled by an **diffusion equation** of the form
<p align=center> 
$\frac{\partial n}{\partial t} = \nabla \cdot \left( D(n) (\nabla n + \epsilon n \nabla n) \right),$
</p>

where $D(n)$ is the particle mobility, $\epsilon$ models how much the ink particle likes to be in the solvent. In this case our free energy needs to be updated to $f=f_1 + \epsilon n \phi$. 
Applying the following transformations $x = \sqrt{\sigma} x^\*$, $t = t^\*/M$ and letting $\alpha := M \sigma$, we obtain

<p align=center>
  $\frac{\partial \phi}{\partial t} = \Delta \phi + \phi - \phi^3 - n \epsilon + \mu  $
<p>
<p align=center>
  $\frac{\partial n}{\partial t} = \frac{1}{\alpha} \cdot \nabla \cdot \left( D(n) \cdot (\nabla n + \epsilon n \nabla n) \right)$
<p>
