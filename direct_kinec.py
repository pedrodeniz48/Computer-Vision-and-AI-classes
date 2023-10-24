import numpy as np
import math

def calc_A(ai, alphai, di, thi):
    
    Rz = np.array([[math.cos(thi), -math.sin(thi), 0, 0],
                 [math.sin(thi), math.cos(thi), 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]);

    Tz = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, di],
                   [0, 0, 0, 1]])

    Tx = np.array([[1, 0, 0, ai],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    Rx = np.array([[1, 0, 0, 0],
                   [0, math.cos(alphai), -math.sin(alphai), 0],
                   [0, math.sin(alphai), math.cos(alphai), 0],
                   [0, 0, 0, 1]])

    Ai = Rz*Tz*Tx*Rx

    print(Ai)

%Dimensiones de robot (cm)
offset = 10;
l1 = 12.9;
l2 = 12.9;

%Número de pasos para interpolaciones
no_interp = 10;

%Puntos a seguir por paso
real_init_pose = [12.5 12.5];
real_final_pose = [10 25];

%Adaptar el offset
init_pose = [real_init_pose(1)-offset real_init_pose(2)];
final_pose = [real_final_pose(1)-offset real_final_pose(2)];

%Interpolaciones
Oyr0 = (init_pose(2))*ones(1, no_interp);
Oyr1 = linspace(init_pose(2), final_pose(2), no_interp);
Oyr = [Oyr0 Oyr1];

Oxr0 = (init_pose(1))*ones(1, no_interp);
Oxr1 = linspace(init_pose(1), final_pose(1), no_interp);
Oxr = [Oxr0 Oxr1];

t = -360:40:360;

sz = size(Oxr);

%Para graficación
Xr = Oxr;
Yr = Oyr;

%Orientación
lado = -1;
q1r = [];
q2r = [];
q3r = [];

figure;
for i = 1:sz(1,2)
    %Cálculo de cinemática inversa
    Qr = inv_kinec_3links_2DOF(Xr(1,i),Yr(1,i),l1,l2,lado);
    q1r = [q1r Qr(1,1)];
    q2r = [q2r Qr(1,2)];
    %q3r = [q3r Qr(1,3)];
    
    %Tablas Denavith-Hartenberg
    a1r=l1; alpha1r=0; d1r=0; th1r = 0 + q1r(1,i);
    a2r=l2; alpha2r=0; d2r=0; th2r = 0 + q2r(1,i);
    
    %Transformaciones
    T1r=calc_A(a1r,alpha1r,d1r,th1r);
    T2r=calc_A(a2r,alpha2r,d2r,th2r);
    
    %Cinemática directa (multiplicación de matrices)
    Tnr=T1r*T2r;

    %Puntos finales de las piernas
    Xrobotr = Tnr(1,4);
    Yrobotr = Tnr(2,4);
    
    %Graficar
    hold on

    plot(0, 0,'sg',Xrobotr,Yrobotr,'or',Tnr(1,4)-(offset*lado),Tnr(2,4),'ob',T1r(1,4),T1r(2,4),'or');

    xlim([-20, 20]);
    ylim([-5, 35]);
    
    grid on

    line([0,T1r(1,4)],[0,T1r(2,4)],'color','red');
    line([T1r(1,4),Tnr(1,4)],[T1r(2,4),Tnr(2,4)],'color','red');
    line([Tnr(1,4)-(offset*lado),Tnr(1,4)],[Tnr(2,4),Tnr(2,4)],'color','red');
    
    hold off
    
    %disp(Oy(i))
    pause(0.05);
    
    if i == sz(1,2)
        break
    end
    
    clf;

end

calc_A(1, 0.5, 0.75, 1)

%Ángulos para motores en grados
ang_rad = [transpose(q1r) transpose(q2r)];% transpose(q3r)];
ang_rad = [ang_rad ang_rad(:,1)+ang_rad(:,2)];
ang_rad(:,1) = ang_rad(:,1)-pi/2;
ang_deg = rad2deg(ang_rad)
ang_deg(:, 1) = ang_deg(:, 1);
ang_deg(:, 2) = ang_deg(:, 2);
ang_deg(:, 3) = ang_deg(:, 3);
%A = abs(round(ang_deg))

%Radianes en orden descendente
rads = deg2rad(ang_deg);