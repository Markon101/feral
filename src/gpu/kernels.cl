// FERAL FP32 reference kernels. No relaxed math; host/device layout is frame-major.
#pragma OPENCL FP_CONTRACT OFF
#define TILE 16
kernel void features(global const float *data, global float *x,
                     uint n, uint block, int pre, float cin, float scale) {
  uint id=get_global_id(0), channels=3*block+4, frames=n/block;
  if(id>=frames*channels)return;
  uint p=id/channels, c=id%channels;
  if(c<block) {
    uint i=p*block+c;float y=data[i],z=data[n+i],m=data[2*n+i];
    float t=data[3*n];float xt=m==1.0f?y:(1.0f-t)*z+t*y;
    x[id]=pre?(m==1.0f?y/scale:xt*cin):xt;
  } else if(c<2*block) {
    uint i=p*block+c-block;float y=data[i];x[id]=(pre?y/scale:y)*data[2*n+i];
  } else if(c<3*block) x[id]=data[2*n+p*block+c-2*block];
  else x[id]=data[3*n+c-3*block];
}
kernel void pool_frames(global const float *x, global float *y,uint frames,uint channels,uint scale) {
  uint id=get_global_id(0), groups=(frames+scale-1)/scale;if(id>=groups*channels)return;
  uint q=id/channels,c=id%channels,count=min(scale,frames-q*scale);float sum=0.0f;
  for(uint r=0;r<count;r++)sum+=x[(q*scale+r)*channels+c]/(float)count;
  y[id]=sum;
}
// Tiled im2col GEMM. Nonlinearity and residual upsampling are fused with output.
kernel void conv_forward(global const float *x,global const float *weights,global const float *skip,
                         global float *z,global float *y,uint frames,uint ci,uint co,uint kernel_size,
                         uint dilation,uint wo,uint bo,uint full_frames,uint scale,int activation,int residual) {
  uint lc=get_local_id(0),lr=get_local_id(1),p=get_group_id(1)*TILE+lr,o=get_group_id(0)*TILE+lc;
  local float xx[TILE][TILE],ww[TILE][TILE];float sum=o<co?weights[bo+o]:0.0f;
  uint reduce=ci*kernel_size;
  for(uint base=0;base<reduce;base+=TILE) {
    uint r=base+lc,k=r/ci,i=r%ci;int q=(int)p+((int)k-(int)(kernel_size/2))*(int)dilation;
    xx[lr][lc]=(p<frames&&r<reduce&&q>=0&&q<(int)frames)?x[(uint)q*ci+i]:0.0f;
    uint out=get_group_id(0)*TILE+lr,ri=base+lc;
    ww[lc][lr]=(out<co&&ri<reduce)?weights[wo+out*reduce+ri]:0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for(uint j=0;j<TILE;j++)sum+=xx[lr][j]*ww[j][lc];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(p<frames&&o<co) {
    float a=activation?tanh(sum):sum;z[p*co+o]=a;
    uint count=min(scale,full_frames-p*scale);
    for(uint r=0;r<count;r++){uint at=(p*scale+r)*co+o;y[at]=residual?skip[at]+0.5f*a:a;}
  }
}
kernel void loss_grad(global const float *prediction,global const float *data,global float *dy,
                      global float *partial,uint n,float unknown,int pre,float cskip,float cout) {
  uint id=get_global_id(0),lane=get_local_id(0);local float scratch[256];float value=0.0f;
  if(id<n) {
    float y=data[id],z=data[n+id],m=data[2*n+id],t=data[3*n];
    float xt=m==1.0f?y:(1.0f-t)*z+t*y;
    float target=pre?(y-z-cskip*xt)/cout:y-z;
    float d=(prediction[id]-target)*(1.0f-m);dy[id]=2.0f*d/unknown;value=d*d/unknown;
  }
  scratch[lane]=value;barrier(CLK_LOCAL_MEM_FENCE);
  for(uint step=128;step>0;step>>=1){if(lane<step)scratch[lane]+=scratch[lane+step];barrier(CLK_LOCAL_MEM_FENCE);}
  if(lane==0)partial[get_group_id(0)]=scratch[0];
}
kernel void activation_grad(global const float *dh,global const float *z,global float *dz,
                            uint full_frames,uint channels,uint scale,int residual) {
  uint id=get_global_id(0),frames=(full_frames+scale-1)/scale;if(id>=frames*channels)return;
  uint q=id/channels,c=id%channels,count=min(scale,full_frames-q*scale);float g=0.0f;
  for(uint r=0;r<count;r++)g+=dh[(q*scale+r)*channels+c];
  float a=z[id];dz[id]=g*(1.0f-a*a)*(residual?0.5f:1.0f);
}
// dW = dZ^T * im2col(X), with a fused bias reduction in column zero.
kernel void weight_grad(global const float *x,global const float *dz,global float *grad,
                        uint frames,uint ci,uint co,uint kernel_size,uint dilation,uint wo,uint bo) {
  uint lc=get_local_id(0),lr=get_local_id(1),o=get_group_id(1)*TILE+lr,r=get_group_id(0)*TILE+lc;
  uint reduce=ci*kernel_size;local float zz[TILE][TILE],xx[TILE][TILE];float sum=0.0f,bias=0.0f;
  for(uint base=0;base<frames;base+=TILE) {
    uint pp=base+lr,oo=get_group_id(1)*TILE+lc;
    zz[lc][lr]=(pp<frames&&oo<co)?dz[pp*co+oo]:0.0f;
    uint k=r/ci,i=r%ci;int q=(int)pp+((int)k-(int)(kernel_size/2))*(int)dilation;
    xx[lr][lc]=(pp<frames&&r<reduce&&q>=0&&q<(int)frames)?x[(uint)q*ci+i]:0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for(uint j=0;j<TILE;j++){sum+=zz[lr][j]*xx[j][lc];if(r==0)bias+=zz[lr][j];}
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(o<co&&r<reduce)grad[wo+o*reduce+r]=sum;
  if(o<co&&r==0)grad[bo+o]=bias;
}
// dX uses reversed kernel order to match CPU's increasing source-frame order.
// Pool transpose and the full-resolution residual skip derivative are fused.
kernel void input_grad(global const float *dz,global const float *weights,global const float *dh_skip,
                       global float *dx,uint frames,uint ci,uint co,uint kernel_size,uint dilation,
                       uint wo,uint full_frames,uint scale,int residual) {
  uint lc=get_local_id(0),lr=get_local_id(1),q=get_group_id(1)*TILE+lr,i=get_group_id(0)*TILE+lc;
  local float zz[TILE][TILE],ww[TILE][TILE];float sum=0.0f;uint reduce=co*kernel_size;
  for(uint base=0;base<reduce;base+=TILE) {
    uint r=base+lc,k=kernel_size-1-r/co,o=r%co;int p=(int)q-((int)k-(int)(kernel_size/2))*(int)dilation;
    zz[lr][lc]=(q<frames&&r<reduce&&p>=0&&p<(int)frames)?dz[(uint)p*co+o]:0.0f;
    uint rr=base+lr,kk=kernel_size-1-rr/co,oo=rr%co;
    ww[lr][lc]=(rr<reduce&&i<ci)?weights[wo+(oo*kernel_size+kk)*ci+i]:0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for(uint j=0;j<TILE;j++)sum+=zz[lr][j]*ww[j][lc];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(q<frames&&i<ci) {
    uint count=min(scale,full_frames-q*scale);
    for(uint r=0;r<count;r++){uint at=(q*scale+r)*ci+i;dx[at]=sum/(float)count+(residual?dh_skip[at]:0.0f);}
  }
}
kernel void statistics(global const float *grad,global const float *loss_parts,global float *stats,
                       uint parameters,uint parts) {
  uint lane=get_local_id(0);local float sums[256],losses[256];local int bad[256];float sum=0.0f,loss=0.0f;int invalid=0;
  for(uint i=lane;i<parameters;i+=256){float g=grad[i];sum+=g*g;invalid|=!isfinite(g);}
  for(uint i=lane;i<parts;i+=256)loss+=loss_parts[i];
  sums[lane]=sum;losses[lane]=loss;bad[lane]=invalid;barrier(CLK_LOCAL_MEM_FENCE);
  for(uint s=128;s>0;s>>=1){if(lane<s){sums[lane]+=sums[lane+s];losses[lane]+=losses[lane+s];bad[lane]|=bad[lane+s];}barrier(CLK_LOCAL_MEM_FENCE);}
  if(lane==0){stats[0]=losses[0];stats[1]=sqrt(sums[0]);stats[2]=1.0f/fmax(1.0f,stats[1]);stats[3]=0.0f;if(bad[0]||!isfinite(stats[0])||!isfinite(stats[1]))stats[3]=1.0f;}
}
kernel void adam_update(global float *weights,global float *m,global float *v,global const float *grad,
                        global float *stats,uint n,float lr,float c1,float c2) {
  // stats[3] is the atomic error flag. Do not concurrently read it nonatomically.
  uint i=get_global_id(0);if(i>=n||!isfinite(stats[0])||!isfinite(stats[1]))return;
  float g=grad[i]*stats[2],mm=0.9f*m[i]+0.1f*g,vv=0.999f*v[i]+0.001f*g*g;
  float p=weights[i]-lr*(mm/c1)/(sqrt(vv/c2)+1e-8f);m[i]=mm;v[i]=vv;weights[i]=p;
  if(!isfinite(p)||!isfinite(mm)||!isfinite(vv))atomic_or((volatile global int*)(stats+3),1);
}
