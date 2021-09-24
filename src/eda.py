import os
import argparse

import pandas as pd
import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator

import streamlit as st

# import matplotlib.pyplot as plt
# import seaborn as sns

# import plotly.express as px

import hvplot.pandas

import holoviews as hv
# from holoviews import opts

from utils import extract_study_phase, format_number_header

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../assets/')
DATA_DIR = '/Volumes/lts4-immuno/data_2021-09-20'

parser = argparse.ArgumentParser(description='graphmel eda')

parser.add_argument('--data', default=DATA_DIR, help='Data directory with 4 csv files.')

try:
    args = parser.parse_args()
except SystemExit as e:
    os._exit(e.code)
    
DATA_DIR = args.data

st.set_page_config(
    layout="centered",
    page_icon="https://pbs.twimg.com/profile_images/1108310620590559232/7fTy3YtS_400x400.png")

lesions_file = 'melanoma_lesion-info_organ-overlap_2021-09-17_anonymized_cleaned_all.csv'
mappings_file = 'melanoma_lesion_mapping_2021-09-20_anonymized.csv'
patients_file = 'melanoma_patient-level_summary_anonymized.csv'
studies_file = 'melanoma_study_level_summary_anonymized.csv'

# Load data into Pandas DataFrames
patients = pd.read_csv(os.path.join(DATA_DIR, patients_file))
studies = pd.read_csv(os.path.join(DATA_DIR, studies_file))
lesions = pd.read_csv(os.path.join(DATA_DIR, lesions_file))
mappings = pd.read_csv(os.path.join(DATA_DIR, mappings_file))

# Short preprocessing -- TODO: create a given preprocessing script
patients['age_at_treatment_start_in_years'] = \
    patients.age_at_treatment_start_in_years.apply(lambda a: 90 if a == '90 or older' else int(a))

# Styles
with open(os.path.join(ASSETS_DIR, "style.css")) as file:
    styles = file.read()
st.write(f"<style> {styles} </style>", unsafe_allow_html=True)

st.title("Graph analytics for immunotherapy response prediction in melanoma")
st.write("Exploratory Data Analytics")

# TODO: reformat this in a function
studies_pp = studies.groupby('gpcr_id').size().mean()
avg_studies_pp = f'{studies_pp:,.2f} per patient on average'

lesions_pp = lesions.groupby(['gpcr_id', 'study_name']).size().mean()
avg_lesions_pp = f'{lesions_pp:,.2f} per study on average'

st.components.v1.html(
    f"""<style> {styles} </style> \
    <div class="container-numbers"> \
        {format_number_header('Patients', patients.shape[0], 'total' )} \
        {format_number_header('Studies', studies.shape[0], avg_studies_pp)} \
        {format_number_header('Lesions', lesions.shape[0], avg_lesions_pp)} \
    </div>"""
)

st.markdown("### Patients")

with st.expander("See analysis"):
    
    age_hist = patients.hvplot.hist('age_at_treatment_start_in_years',
                                    xlabel='Age at treatment start (years)',
                                    ylabel='Amount of patients', width=900)
    
    treatment_duration_hist = patients.hvplot.hist('duration_treatment_in_days',
                                                   by='death_event_observed',
                                                   subplots=True, width=450, legend='top',
                                                   xlabel='Treatment duration (days)',
                                                   ylabel='Amount of patients')
    
    duration_survival_scat = patients.hvplot.scatter(x='duration_treatment_in_days',
                                                     y='survival_in_days', width=900,
                                                     by='death_event_observed', legend='top',
                                                     xlabel='Treatment duration (days)',
                                                     ylabel='Survival (days)')
    
    time, s_prob = kaplan_meier_estimator(patients.death_event_observed, patients.survival_in_days)
    km_df = pd.DataFrame(np.array([time, s_prob]).T, columns=['time', 'survival_probability'])

    survival_km = km_df.hvplot.step(x='time', y='survival_probability',
                                    title='Kaplan-Meier survival curve', width=900,
                                    xlabel='Time (days)', ylabel='Survival probability')
    
    st.bokeh_chart(
        hv.render(
            (
                age_hist + treatment_duration_hist + duration_survival_scat + survival_km
            ).cols(1), backend='bokeh'
        )
    )

st.markdown("### Studies")

with st.expander("See analysis"):
    
    patients = st.multiselect('Select patient identifier',
                              list(studies.gpcr_id.unique()),
                              studies.gpcr_id.unique()[:5])
    
    lines = []
    
    for patient in patients:
        lines.append(studies[studies.gpcr_id == patient].hvplot.line(x='study_name',
                                                                     y='is_malignant',
                                                                     width=900, xlabel='Study',
                                                                     ylabel='Malignant lesions'))
        
    st.bokeh_chart(
        hv.render(
            hv.Overlay(lines), backend='bokeh'
        )
    )

st.markdown("### Lesions")

with st.expander("See analysis"):

    lr = lesions.groupby(['pars_region_petct',
                          'pars_classification_petct']).size().to_frame('count').reset_index()

    lr_hm = lr.hvplot.heatmap(y='pars_region_petct', x='pars_classification_petct', C='count',
                              ylabel='Region', xlabel='Classification', colorbar=False,
                              cmap='viridis',
                              title='Region classification', width=450, height=500)
    
    location = lesions.groupby(['pars_region_petct',
                                'pars_laterality_petct']).size().to_frame('count').reset_index()

    loc_hm = location.hvplot.heatmap(x='pars_laterality_petct', y='pars_region_petct', C='count',
                                     xlabel='Laterality', ylabel='Region', width=450, height=500,
                                     title='Lesion laterality',
                                     cmap='viridis', yaxis='bare')
    
    st.bokeh_chart(
        hv.render(
            (lr_hm + loc_hm).cols(2).opts(shared_axes=True), backend='bokeh'
        )
    )
    
    organ_count = lesions.groupby(['assigned_organ',
                                   'is_malignant']).size().to_frame('count').reset_index()
    organ_count['is_malignant'] = \
        organ_count.is_malignant.apply(lambda b: 'malignant' if b else 'benign')

    organs = hv.Dimension('assigned_organ', label='Assigned organ')
    count = hv.Dimension('count', label='Count', unit='lesions')
    malignant = hv.Dimension('is_malignant', label='Malignant')

    organ_bars = hv.Bars(organ_count, kdims=['assigned_organ', 'is_malignant'], vdims='count') \
        .opts(stacked=True, width=900, xrotation=15, xlabel='Assigned organ', ylabel='Lesions') \
        .sort(by='count', reverse=True)
    
    scan_pp = lesions.groupby(['gpcr_id', 'study_name']).size().to_frame('lesions').reset_index()
    scan_pp['study_name'] = scan_pp.study_name.apply(extract_study_phase)
    scan = hv.Dimension('study_name', label='Scan w.r.t. treatment')
    lesion = hv.Dimension('lesions', label='Lesions')

    spp = hv.BoxWhisker(scan_pp, kdims=scan, vdims=lesion).opts(width=900)

    patients_ps = scan_pp[['study_name', 'gpcr_id']] \
        .groupby('study_name').size().to_frame('patients').reset_index()
    patients = hv.Dimension('patients', label='Patients')

    pps = hv.Bars(patients_ps, kdims=scan, vdims=patients).opts(width=900)
    
    st.bokeh_chart(
        hv.render(
            (organ_bars + pps + spp.sort()).cols(1), backend='bokeh'
        )
    )

st.markdown("### Lesion mapping""")

with st.expander("See analysis"):
    st.write("Coming soon!")
