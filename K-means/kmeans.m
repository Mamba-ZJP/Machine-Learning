clc
clear
tic
RGB= imread ('D:\Picture\С����.jpg'); 
img=rgb2gray(RGB);
[m,n]=size(img);
subplot(2,2,1),imshow(img);title('ԭͼ��')
subplot(2,2,2),imhist(img);title('ԭͼ��ĻҶ�ֱ��ͼ')
hold off;
img=double(img);
for i=1:200
    c1(1)=25;
    c2(1)=125;
    c3(1)=200;
    r=abs(img-c1(i));
    g=abs(img-c2(i));
    b=abs(img-c3(i));
    r_g=r-g;
    g_b=g-b;
    r_b=r-b;
    n_r=find(r_g<=0&r_b<=0);
    n_g=find(r_g>0&g_b<=0);
    n_b=find(g_b>0&r_b>0);
    i=i+1;
    c1(i)=sum(img(n_r))/length(n_r);
    c2(i)=sum(img(n_g))/length(n_g);
    c3(i)=sum(img(n_b))/length(n_b);
    d1(i)=abs(c1(i)-c1(i-1));
    d2(i)=abs(c2(i)-c2(i-1));
    d3(i)=abs(c3(i)-c3(i-1));
    if d1(i)<=0.001&&d2(i)<=0.001&&d3(i)<=0.001
        R=c1(i);
        G=c2(i);
        B=c3(i);
        k=i; 
        break;
    end
end
img=uint8(img);
img(find(img<R))=0;
img(find(img>R&img<G))=128;
img(find(img>G))=255;
toc
subplot(2,2,3),imshow(img);title('������ͼ��') 
subplot(2,2,4),imhist(img);title('������ͼ��ֱ��ͼ') 
