
clear all;

% Init control polygon
f = fopen('controlpoints.txt','r');
formatSpec = '%f';
dim = [2 4];
cPoly = fscanf(f,formatSpec,dim)';
axis([0 1 0 1]);
plot(cPoly(:,1), cPoly(:,2),'b-s','MarkerFaceColor','b');
hold on;

% Make room for curve
stepSize = 0.01;
num_c = round(1/stepSize)+1;
c = zeros(num_c, 2);

% Iterate over curve
for i = 0:stepSize:1
    c(round(i*(1/stepSize))+1,:) = deCasteljau(cPoly, i);
end

plot(c(:,1),c(:,2),'r','LineWidth',2);

% Generate u
step_u = pi/30;
u = 0:step_u:2*pi;
num_u = round(2*pi/step_u)+1;

% Revolving the curve around Z axis
X1 = c(:,1)*cos(u);
Y1 = c(:,1)*sin(u);
Z1 = c(:,2)*ones(1,num_u);

% Revolving the curve along the start-end line
% calculate rotation angle from start-end axis to z-axis
dy = c(1,1)-c(num_c,1);
dz = c(1,2)-c(num_c,2);
sin_theta = dy/sqrt(dy*dy + dz*dz);
cos_theta = dz/sqrt(dy*dy + dz*dz);

% Compute transformation matrix
temp1 = zeros(num_c,1);
temp2 = ones(num_c,1);
concated_c = cat(2,temp1,c,temp2);

rotate_mat = [[1,     0,      0,        0];
              [0,cos_theta, -sin_theta, 0];
              [0,sin_theta, cos_theta,  0];
              [0,     0,      0,        1]];
          
translate_mat = [[1, 0, 0, -concated_c(1,1)];
                 [0, 1, 0, -concated_c(1,2)];
                 [0, 0, 1, -concated_c(1,3)];
                 [0, 0, 0,               1]];
             
tranform_mat = rotate_mat * translate_mat;

% Transform all points
transformed_c =  concated_c * tranform_mat';
% plot(transformed_c(:,2),transformed_c(:,3),'r','LineWidth',2);

% Revolving the curve around Z axis
x = transformed_c(:,2)*cos(u);
y = transformed_c(:,2)*sin(u);
z = transformed_c(:,3)*ones(1,num_u);

new_points = [x(:),y(:),z(:),ones(size(x(:)))];
new_points = new_points * (inv(tranform_mat))';
X2 = reshape(new_points(:,1),size(x));
Y2 = reshape(new_points(:,2),size(x));
Z2 = reshape(new_points(:,3),size(x));


% % Plot surface
% Revolve along z-axis
figure
plot3(zeros(size(cPoly(:,1))), cPoly(:,1), cPoly(:,2),'b-s','MarkerFaceColor','b');
hold on;
plot3(concated_c(:,1),concated_c(:,2),concated_c(:,3),'r','LineWidth',2);
surf(X1,Y1,Z1)
xlabel('X')
ylabel('Y')
zlabel('Z')
view(130,30)

% Revolve along start-end axis
figure
plot3(zeros(size(cPoly(:,1))), cPoly(:,1), cPoly(:,2),'b-s','MarkerFaceColor','b');
hold on;
plot3(concated_c(:,1),concated_c(:,2),concated_c(:,3),'r','LineWidth',2);
surf(X2,Y2,Z2)
xlabel('X')
ylabel('Y')
zlabel('Z')
view(130,30)
% mesh(X1, Y1, Z1);



