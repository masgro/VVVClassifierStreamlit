from typing import Union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from astropy.table import Table
import flask
from joblib import dump, load
import os
import base64
 
st.title('VVVx Classification')
st.header("Random Forest clasifier")
st.subheader("Params")
#st.text("Esto es texto")
#st.latex("y = x^2")
#st.code("if a == 1:\n    print(a)", language="python")
#st.code("var a = 1;", language="javascript")
#st.markdown("Esto es **texto** usando *Markdown*")

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return(href)

def proccess_fits(_data):
  names = [name for name in _data.colnames if len(_data[name].shape) <= 1]
  data = _data[names].to_pandas()
  data[['MAG_APER_KS_0','MAG_APER_KS_1','MAG_APER_KS_2','MAG_APER_KS_3']] = pd.DataFrame(_data['MAG_APER'].tolist())
  data[['MAGERR_APER_KS_0','MAGERR_APER_KS_1','MAGERR_APER_KS_2','MAGERR_APER_KS_3']] = pd.DataFrame(_data['MAGERR_APER'].tolist())
  data[['FLUX_RADIUS_0','FLUX_RADIUS_1','FLUX_RADIUS_2']] = pd.DataFrame(_data['FLUX_RADIUS'].tolist())
  data.rename(columns = {'MAG_APER_KS_0': 'MAG_APER_KS','MAG_APER_KS_0_CORR': 'MAG_APER_KS_CORR', 'MAGERR_APER_KS_0': 'MAGERR_APER_KS'}, inplace = True)
  data.rename(columns = {'MAG_APER_KS': 'MAG_APER'}, inplace = True)
  data.rename(columns = {'MAG_APER_J_0': 'MAG_APER_J', 'MAG_APER_J_0_CORR': 'MAG_APER_J_CORR', 'MAGERR_APER_J_0': 'MAGERR_APER_J'}, inplace = True)
  data.rename(columns = {'MAG_APER_H_0': 'MAG_APER_H', 'MAG_APER_H_0_CORR': 'MAG_APER_H_CORR', 'MAGERR_APER_H_0': 'MAGERR_APER_H'}, inplace = True)
  return(data)

def proccess_csv(data):
  data[['FLUX_RADIUS_0','FLUX_RADIUS_1','FLUX_RADIUS_2']] = data['FLUX_RADIUS'].apply(lambda x: x.replace('(','').replace(')','')).str.split(',',expand=True).astype(float)
  return(data)

columns = ['MAG_PSF','MAG_AUTO','MAG_APER','MAG_MODEL','SPREAD_MODEL',
           'AMODEL_IMAGE','BMODEL_IMAGE','ELONGATION','ELLIPTICITY',
           'A_IMAGE','B_IMAGE','SPHEROID_SERSICN','MU_MAX', 'CLASS_STAR',
           'MAG_APER_H','MAG_PSF_H','MAG_APER_J', 'MAG_PSF_J', 
           'A_v', 'A_Ks', 'A_H', 'A_J',
           'MAG_APER_KS_CORR','MAG_PSF_KS_CORR','MAG_APER_H_CORR',
           'MAG_PSF_H_CORR','MAG_APER_J_CORR', 'MAG_PSF_J_CORR', 
           'FLUX_RADIUS_0', 'FLUX_RADIUS_1', 'FLUX_RADIUS_2',
           'colorJH','colorHKs','colorJKs']

with open(f'pipeline.joblib', 'rb') as f:
    model = load(f)

filetype = st.radio("What's your favorite movie genre",('Fits', 'CSV'))

test = "False"
if filetype == "CSV":
  test = st.radio("Realizar el test",("False","True"))

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    if filetype == 'Fits':
      data = Table.read(uploaded_file, format='fits')
      data = proccess_fits(data)
    else:
      data = pd.read_csv(uploaded_file,delim_whitespace=True)
      data = proccess_csv(data)

    if test == "True": 
      if "GALAXY" not in data.columns:
        st.error('Dataframe must contain boolean column named "GALAXY"')
        st.stop()
      else:
        y = data['GALAXY']

    data['colorJH'] = data['MAG_APER_J_CORR'] - data['MAG_APER_H_CORR']
    data['colorHKs'] = data['MAG_APER_H_CORR'] - data['MAG_APER_KS_CORR']
    data['colorJKs'] = data['MAG_APER_J_CORR'] - data['MAG_APER_KS_CORR']

    for col in columns:
      if col not in data.columns:
        st.error('Dataframe does not contain column %s' % col)
        st.stop()
        
    X = data[columns]
    predictions = model.predict(X)
    
    ngals = sum(predictions == 1)
    nothers = sum(predictions == 0)

    st.markdown("10 random-rows")
    st.dataframe(X.sample(10))

    st.header("Results")
    st.markdown("Ngals: %d" % ngals)
    st.markdown("Nothers: %d" % nothers)

    if test == "True":
      np.set_printoptions(precision=2)

      y_true, y_pred = y, predictions

      TP = (y_true == y_pred) & (y_true == 0)
      FP = (y_true != y_pred) & (y_true == 0)
      TN = (y_true == y_pred) & (y_true == 1)
      FN = (y_true != y_pred) & (y_true == 1)
      
      totalP = sum(y_true == 0)
      totalN = sum(y_true == 1)
      #confusion = np.array([[sum(TP),sum(FP)],[sum(FN),sum(TN)]])
      norm_confusion = np.array([[sum(TP)/totalP,sum(FP)/totalP],[sum(FN)/totalN,sum(TN)/totalN]])
      #
      #st.text(confusion)
      #st.text(norm_confusion)
      
      fig,ax = plt.subplots(1,1,figsize=(10,10))
      ax.matshow(norm_confusion,cmap='Purples',vmin=0.0,vmax=1.0)
      ax.text(0,0,"%3d\n%4.2f" % (sum(TP),sum(TP)/totalP),backgroundcolor='white',ha="center",size='large')
      ax.text(1,0,"%3d\n%4.2f" % (sum(FP),sum(FP)/totalP),backgroundcolor='white',ha="center",size='large')
      ax.text(0,1,"%3d\n%4.2f" % (sum(FN),sum(FN)/totalN),backgroundcolor='white',ha="center",size='large')
      ax.text(1,1,"%3d\n%4.2f" % (sum(TN),sum(TN)/totalN),backgroundcolor='white',ha="center",size='large')

      st.pyplot(fig)

      st.markdown(get_table_download_link(pd.DataFrame(predictions)), unsafe_allow_html=True)
