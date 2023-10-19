# PINN
physics-informed-neural-network
This code was refered from this url:[https://www.bilibili.com/video/BV1MP41187M3/?spm_id_from=333.999.0.0&vd_source=27778d8c1243c6ab95ed0c77018e2063]
Use the governing equation 
$$\frac{\partial^2 u}{\partial^2 x} - \frac{\partial^4 u}{\partial^4 y} = (2-x^2)e^{-y}$$
The initial condition and boundary condition is 
$$\frac{\partial^2 u(x,0)}{\partial^2 y} = x^2,\qquad \frac{\partial^2 u(x,1)}{\partial^2 y} = x^2$$
$$u(x,0)=x^2,\qquad u(x,1)=\frac{x^2}{e}$$
$$u(0,y)=0,\qquad u(1,y)=e^{-y}$$
