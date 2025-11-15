function piston_dispersive_pdepe
% modelo pistão-dispersivo
% dC/dtheta = (1/Pe)*d2C/dx2 - dC/dx
% dominio: x em [0,1], theta >= 0
% CI: C(0,x)=0
%
% fronteiras:
% (i) x=0 danckwerts  | x=1 Neumann
% (ii) x=0 dirichlet  | x=1 neumann
%
% objetivo: comparar C(theta, x=1) para Pe = 5, 50, 100

% malha em x e em theta (tempo)
x = linspace(0,1,201);
t = linspace(0,2,1001);
Pe_vals = [5 50 100];

% se (i): danckwerts na entrada, neumann na saida
figure; % figura separada
hold on;
for k = 1:numel(Pe_vals)
    Pe = Pe_vals(k);
    % pdepe precisa destas 4 coisas: m, pdefun, icfun, bcfun, e as malhas
    sol = pdepe(0, @(x,t,u,dudx)pdefun_local(x,t,u,dudx,Pe), ...
                   @icfun_local, ...
                   @(xl,ul,xr,ur,t)bc_i_local(xl,ul,xr,ur,t,Pe), ...
                   x, t);
    % C(theta, x=1) - ultima coluna no eixo x
    C_out = sol(:,end,1);
    plot(t, C_out, 'DisplayName', ['Pe = ' num2str(Pe)]);
end
xlabel('\theta');
ylabel('C(\theta,1)');
title('caso (i): danckwerts na entrada, neumann na saida');
legend; grid on;

% se (ii): dirichlet na entrada, neumann na saida 
figure; % outra figura
hold on;
for k = 1:numel(Pe_vals)
    Pe = Pe_vals(k);
    sol = pdepe(0, @(x,t,u,dudx)pdefun_local(x,t,u,dudx,Pe), ...
                   @icfun_local, ...
                   @(xl,ul,xr,ur,t)bc_ii_local(xl,ul,xr,ur,t,Pe), ...
                   x, t);
    C_out = sol(:,end,1);
    plot(t, C_out, 'DisplayName', ['Pe = ' num2str(Pe)]);
end
xlabel('\theta');
ylabel('C(\theta,1)');
title('caso (ii): dirichlet na entrada, neumann na saida ');
legend; grid on;

end

% funcoes locais

function [c,f,s] = pdefun_local(x,t,u,dudx,Pe)
% aqui escolhi:
% c = 1
% f = (1/Pe)*dC/dx  (assim a derivada em x de f dá o termo difusivo)
% s = -dC/dx        (termo convectivo)
c = 1;
f = (1/Pe)*dudx;
s = -dudx;
end

function u0 = icfun_local(x)
% condicao inicial: tudo a zero
u0 = 0;
end

function [pl,ql,pr,qr] = bc_i_local(xl,ul,xr,ur,t,Pe)
% CASO (i)
% x=0 (entrada) Danckwerts: Cin - C + (1/Pe) dC/dx = 0
% em pdepe: p + q*f = 0 -> basta p = Cin - C, q = 1 (porque f já tem (1/Pe)*dC/dx)
Cin = 1;      % degrau unitario
pl = (Cin - ul);
ql = 1;
% x=1 (saida) Neumann: dC/dx = 0  -> f = 0
pr = 0;
qr = 1;
end

function [pl,ql,pr,qr] = bc_ii_local(xl,ul,xr,ur,t,Pe)
% CASO (ii)
% x=0 Dirichlet: C = 1 -> p = C - 1, q = 0
pl = (ul - 1);
ql = 0;
% x=1 Neumann: dC/dx = 0 -> f = 0
pr = 0;
qr = 1;
end


%fontes: material cedido pelo docente 
%https://www.mathworks.com/help/matlab/ref/pdepe.html
%http://www.ece.northwestern.edu/local-apps/matlabhelp/techdoc/ref/pdepe.html
