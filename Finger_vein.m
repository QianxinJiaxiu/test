clear;
clc;
close all;

imgPath = 'C:\Users\QianXin\Documents\MATLAB\�����Ӿ�\128\2\'; % ͼ���·��
imgDir  = dir([imgPath '*.bmp']); % ��������bmp��ʽ�ļ�
for i = 1:length(imgDir)         % �����ṹ��Ϳ���һһ����ͼƬ��
    I = imread([imgPath imgDir(i).name]); %��ȡÿ��ͼƬ
    
    I=rgb2gray(I);
    I = imcrop(I, [20, 50, 640, 370]);%�������ಿ��

    I1=adapthisteq(I);%�Աȶ����Ƶ�����Ӧֱ��ͼ����
    
    %��ȡ��Ե
    I2=edge(I,'sobel',0.05);
    [h,w]=size(I2);
    center=round(h/2);
    for i1=1:w
        temp_index=find(I2(:,i1));
        n=0;
        for j=1:length(temp_index)
            if temp_index(j)<center
                
                n=n+1;
            end
        end
       
        up=n;
        if up==0 %ͼ���ϲ������㣨��Ե�㣩
            if i1==1
                index(i1,1)=index(20,1);
            else
                index(i1,1)=index(i1-1,1);
            end
        else
            index(i1,1)=temp_index(up);%up
        end
        if up==length(temp_index) %ͼ���²������㣨��Ե�㣩
            index(i1,2)=index(i1-1,2);
        else
            down=up+1;
            index(i1,2)=temp_index(down);%down
        end   

    end
    I3=zeros(h,w);
    for k=1:w
        I3(index(k,1),k)=1;
        I3(index(k,2),k)=1;
    end

    %����
    for i2=1:w
        x1(i2)=floor((index(i2,1)+index(i2,2))/2);
        y1(i2)=i2;
        I3(x1(i2),i2)=1;
    end

    %��С���˷�ƥ��б��
    p=polyfit(y1,x1,1);%ע��ͼ������ϵ
    a=p(1);
    b = atan(a);
    I4=imrotate(I1,b*180/pi,'bilinear', 'crop');

    %��ȡROI�����ұ߽�
    I4=imcrop(I4, [20, 10, w-40,h-20]);%��ת���ж���ı�Ե������ȡ��Եʱ�ǲ���Ҫ������
    [h,w]=size(I4);
    center=round(h/2);
    for i3=1:w  
        x(i3)=i3;
        gray(i3)=I4(center,i3);%���ߵĻҶ�ֵ
    end

    %����peak��x��20~560֮��
    step=1;
    pos=21;
    c=1;
    while(pos<560)
        if((gray(pos)>=gray(pos+step)) && (gray(pos)>=gray(pos-step)))
            if((gray(pos)>=gray(pos+20)) && (gray(pos)>=gray(pos-20)))%����
                peaks(c)=pos;
                c=c+1;
            end
        end
        pos=pos+step;
    end
    peak1=1;
    peak2=round(w/2);
    for i4=1:length(peaks)
        if(peaks(i4)<round(w/2))
            if gray(peaks(i4))>=gray(peak1)
                peak1=peaks(i4);
            end
        else
            if gray(peaks(i4))>=gray(peak2)
                peak2=peaks(i4);
            end
        end
    end
    
    %��ȡ��ת���ͼ���Ե��ȷ��ROI�����½�
    I5=edge(I4,'sobel',0.05);
    [h,w]=size(I5);
    center=round(h/2);
    for i5=1:w
        temp_index=find(I5(:,i5));
        n=0;
        for j=1:length(temp_index)
            if temp_index(j)<center
                %up_index(n)=center-temp_index(j,1);
                n=n+1;
            end
        end
        %up=length(up_index);
        %up_index=[];
        up=n;
        if up==0
            if i5==1
                index(i5,1)=index(20,1);%��ֹindex(0,1),��Ч������
            else
                index(i5,1)=index(i5-1,1);
            end            
        else
            index(i5,1)=temp_index(up);%up
        end
        if up==length(temp_index)
            index(i5,2)=index(i5-1,2);
        else
            down=up+1;
            index(i5,2)=temp_index(down);%down
        end   

    end
    
    U=sort(index(:,1),'descend');%�������򣬲�ȡ��ֵ�㣬�˳�����
    upedge=U(50);
    D=sort(index(:,2));
    downedge=D(30);
    I6 = imcrop(I4, [peak1, upedge, peak2-peak1,downedge-upedge]);
     
    
    %������ȡ 
    v=floor(size(I6)/4);
    features =extractLBPFeatures(I6, 'Upright',true,'cellsize',v);
    DV=length(features);
    langmuda=0.01;
    vectorlangmuda=langmuda*ones(1,DV);
    features2=(features*16+vectorlangmuda)*(1/(norm(features,1)+DV*langmuda));
    F(:,i)=features2;
    
end

figure
subplot(3,2,1);
imshow(I);
title('ԭͼ');
subplot(3,2,2);
imshow(I1);
title('�Աȶ����Ƶ�����Ӧֱ��ͼ���⻯')
subplot(3,2,3);
imshow(I2);
title('sobel���ӱ�Ե���')
subplot(3,2,4);
imshow(I3);
title('��ȡ��Ե��������')
subplot(3,2,5);
imshow(I4);
title('��ת��');
subplot(3,2,6);
imshow(I6);
title('ROI');

%ƥ��
for o=1:length(imgDir)
    for u=1:length(imgDir)
        lo=reshape(F(:,o),1,length(F(:,o)));
        lu=reshape(F(:,u),1,length(F(:,u)));
        for t =1:length(lo)
            Guance(o,u)=sum((lo-lu)*(log2(reshape(lo,length(lo),1))-log2(reshape(lu,length(lu),1))));
        end     
    end
end

%��һ��
for o1=1:length(imgDir)
    for u1=1:length(imgDir)
        RGuance(o1,u1)=(Guance(o1,u1)-min(min(Guance)))/(max(max(Guance))-min(min(Guance)));     
    end
end
figure
subplot(1,2,1)
imshow(Guance)
title('��ȡ��ƥ������ֵͼ');
subplot(1,2,2)
imshow(RGuance)
title('��һ����ƥ������ֵͼ');

LN1=reshape(RGuance(1:10,1:10),100,1);
LN2=reshape(RGuance(11:20,11:20),100,1);
LN3=reshape(RGuance(21:30,21:30),100,1);
LN4=reshape(RGuance(31:40,31:40),100,1);
LN1(find(LN1==0))=[];%ɾ��ԭͼ�Լ����Լ�ƥ��
LN2(find(LN2==0))=[];
LN3(find(LN3==0))=[];
LN4(find(LN4==0))=[];

class_in=[ LN1;LN2;LN3;LN4];
figure
subplot(1,2,1);
cihist=histogram(class_in);
title('����ֱ��ͼ�ֲ�')
LW1=reshape(RGuance(1:10,11:40),300,1);
LW2=reshape(RGuance(11:20,1:10),100,1);
LW3=reshape(RGuance(11:20,21:40),200,1);
LW4=reshape(RGuance(21:30,1:20),200,1);
LW5=reshape(RGuance(21:30,31:40),100,1);
LW6=reshape(RGuance(31:40,1:30),300,1);
class_out=[LW1;LW2;LW3;LW4;LW5;LW6];

z = cihist.Values/2;
x = cihist.BinEdges + 0.5*cihist.BinWidth;
x = x(1:(length(x)-1));%bin�߱�����һ��ֵ
subplot(1,2,2);
cohist=histogram(class_out);
title('���ֱ��ͼ�ֲ�')
z1 = cohist.Values/2;
x1 = cohist.BinEdges + 0.5*cohist.BinWidth;
x1 = x1(1:(length(x1)-1));%bin�߱�����һ��ֵ

figure
plot(x,z);
hold on    
plot(x1,z1);
grid on
xlabel('ƥ�䲢��һ��֮��ľ���'); 
ylabel('Ƶ��'); 
legend('���ھ���ֲ�','������ֲ�')
title('�������ھ���������ֲ�ͼ')
