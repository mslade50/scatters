import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import plotly.express as px
import seaborn as sns
import numpy as np
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import datetime as dt
from datetime import date
from datetime import timedelta
import yfinance as yf
from pandas_datareader import data as pdr
from matplotlib import cm
import plotly.io as pio
from io import BytesIO
import requests
import streamlit as st

##convert matplotlib seismic color scale to plotly compatible 
def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale

seismic_cmap= plt.cm.get_cmap('seismic')
seismic = matplotlib_to_plotly(seismic_cmap, 255)
day=date.today()

##read in data to create dataframes.....All files currently housed in sublime misc so will need to adjust if moved. These lines based on old seasonal model
# model_df=pd.read_excel("MDM.personaltweaks.xlsx")
# fwd_df=pd.read_excel("MDM.personaltweaks.xlsx",sheet_name='Python Fwds')
# df = pd.read_csv('MDM.personaltweaks_python.csv')


url_base = 'https://raw.githubusercontent.com/mslade50/Multi_page/main/'

# Download and read the Excel file
excel_url = url_base + 'MDM.personaltweaks_2.xlsx'
response = requests.get(excel_url)
response.raise_for_status()  # Raise an exception if there's an HTTP error
excel_data = BytesIO(response.content)

with pd.ExcelFile(excel_data) as xls:
    model_df = pd.read_excel(xls)
    fwd_df = pd.read_excel(xls, sheet_name='Python Fwds')

# Read the CSV file
csv_url = url_base + 'MDM.personaltweaks_python_2.csv'
df = pd.read_csv(csv_url)

###need to get the most recent values for your predictors. BC they are in excel sheets with formulas extending further down, we need to find a way to
###get the last 'real' value, so after inputting the df we forward fill the necessary columns with our actual data, replacing Nan vals and then use the "last valid index to get the last value"
today=model_df.last_valid_index()
model_df.Seasonals.ffill(inplace=True)
model_df['NPD 5 SMA'].ffill(inplace=True)
model_df['5d Trailing'].ffill(inplace=True)
model_df['21d Trailing'].ffill(inplace=True)
seasonal_today=model_df['Seasonals'][today]/10
trailing_5_today=model_df['5d Trailing'][today]/10
trailing_21_today=model_df['21d Trailing'][today]/10
NPD_today=model_df['NPD 5 SMA'][today]

###same process as above but for the fwd df
today2=fwd_df.last_valid_index()
fwd_df['21dr_rnk_up1'].ffill(inplace=True)
fwd_df['21dr_rnk_down1'].ffill(inplace=True)
fwd_df['21dr_rnk_flat'].ffill(inplace=True)
fwd_df['5dr_rnk_up1'].ffill(inplace=True)
fwd_df['5dr_rnk_down1'].ffill(inplace=True)
fwd_df['5dr_rnk_flat'].ffill(inplace=True)
fwd_df['seasonals_tm'].ffill(inplace=True)
fwd_df['seasonals_fwd_5'].ffill(inplace=True)
rnk_21_up1=(fwd_df['21dr_rnk_up1'][today2]/10).round(2)
rnk_21_down1=(fwd_df['21dr_rnk_down1'][today2]/10).round(2)
rnk_21_flat=(fwd_df['21dr_rnk_flat'][today2]/10).round(2)
rnk_5_up1=(fwd_df['5dr_rnk_up1'][today2]/10).round(2)
rnk_5_down1=(fwd_df['5dr_rnk_down1'][today2]/10).round(2)
rnk_5_flat=(fwd_df['5dr_rnk_flat'][today2]/10).round(2)
seasonal_tm=fwd_df['seasonals_tm'][today2]/10
seasonal_fwd_5=fwd_df['seasonals_fwd_5'][today2]/10

#Non filtered data for heatmaps, has to be converted to lists to be used for the heatmap plots.
npd=df_seasonals['NPD 5 SMA'].values.tolist()
seasonals=df_seasonals['Seasonal_rank'].values.tolist()
trailing_5=df_seasonals['5dt'].values.tolist()
trailing_21=df_seasonals['21dt'].values.tolist()
fwd_1=df_seasonals['1d FWD%'].values.tolist()
fwd_5=df_seasonals['5d FWD%'].values.tolist()
fwd_21=df_seasonals['21d FWD%'].values.tolist()
fwd_10=df_seasonals['10d FWD%'].values.tolist()
fwd_63=df_seasonals['63d FWD%'].values.tolist()
fwd_126=df_seasonals['126d FWD%'].values.tolist()
fwd_252=df_seasonals['252d FWD%'].values.tolist()
fwd_VIX_5=df_seasonals['5d FWD VIX'].values.tolist()
fwd_VIX_21=df_seasonals['21d FWD VIX'].values.tolist()

###annotation for plotly charts
# note=f'Forecast for {day}<br>Black crosshairs are current positioning<br>Gold line is where seasonal rank will be tomorrow ({seasonal_tm*10})<br>If SPX goes up by 1% tomorrow then 21d returns rank will be {rnk_21_up1*10} and 5d returns rank will be {(rnk_5_up1*10).round(1)}<br>If SPX goes down by 1% tomorrow then 21d returns rank will be {(rnk_21_down1*10).round(1)} and 5d returns rank will be {(rnk_5_down1*10).round(1)}'
note=f'Forecast for {day}<br>Black crosshairs are positioning as of yesterday close<br>Circles are where positioning will be at today close given a 1% move up (green) 1% move down (red) or 0% move (grey)<br>Gold circle is where seasonal rank will be a week from today'

##NPD filtered DFs
df_low_NPD=df_seasonals[df_seasonals['NPD 5 SMA']<-6.75]

df_low_mid_NPD=df_seasonals[df_seasonals['NPD 5 SMA']>-10]
df_low_mid_NPD=df_low_mid_NPD[df_low_mid_NPD['NPD 5 SMA']<-5]

df_mid_NPD=df_seasonals[df_seasonals['NPD 5 SMA']>-5]
df_mid_NPD=df_mid_NPD[df_mid_NPD['NPD 5 SMA']<5]

df_high_NPD=df_seasonals[df_seasonals['NPD 5 SMA']>-6.75]

df_npd_today=df_seasonals[df_seasonals['NPD 5 SMA']>NPD_today-2]
df_npd_today=df_npd_today[df_npd_today['NPD 5 SMA']<NPD_today+2]
df_seasonals_today=df_npd_today

# # Create an Excel writer object for the workbook
# writer = pd.ExcelWriter('miscc_1.xlsx', engine='xlsxwriter')

# # Write each dataframe to a separate tab in the workbook
# df_low_NPD.to_excel(writer, sheet_name='Tab1')
# df_high_NPD.to_excel(writer, sheet_name='Tab2')
# df_seasonals.to_excel(writer, sheet_name='Tab3')

# # Save and close the workbook
# writer.save()


###Low NPD filtered heatmap data
seasonals_low_NPD=df_low_NPD['Seasonal_rank'].values.tolist()
trailing_5_low_NPD=df_low_NPD['5dt'].values.tolist()
trailing_21_low_NPD=df_low_NPD['21dt'].values.tolist()
fwd_1_low_NPD=df_low_NPD['1d FWD%'].values.tolist()
fwd_5_low_NPD=df_low_NPD['5d FWD%'].values.tolist()
fwd_10_low_NPD=df_low_NPD['10d FWD%'].values.tolist()
fwd_21_low_NPD=df_low_NPD['21d FWD%'].values.tolist()
fwd_63_low_NPD=df_low_NPD['63d FWD%'].values.tolist()
fwd_126_low_NPD=df_low_NPD['126d FWD%'].values.tolist()
fwd_252_low_NPD=df_low_NPD['252d FWD%'].values.tolist()
fwd_VIX_5_low_NPD=df_low_NPD['5d FWD VIX'].values.tolist()
fwd_VIX_21_low_NPD=df_low_NPD['21d FWD VIX'].values.tolist()

###Low-Mid NPD filtered heatmap data
seasonals_low_mid_NPD=df_low_mid_NPD['Seasonal_rank'].values.tolist()
trailing_5_low_mid_NPD=df_low_mid_NPD['5dt'].values.tolist()
trailing_21_low_mid_NPD=df_low_mid_NPD['21dt'].values.tolist()
fwd_1_low_mid_NPD=df_low_mid_NPD['1d FWD%'].values.tolist()
fwd_5_low_mid_NPD=df_low_mid_NPD['5d FWD%'].values.tolist()
fwd_10_low_mid_NPD=df_low_mid_NPD['10d FWD%'].values.tolist()
fwd_21_low_mid_NPD=df_low_mid_NPD['21d FWD%'].values.tolist()
fwd_63_low_mid_NPD=df_low_mid_NPD['63d FWD%'].values.tolist()
fwd_126_low_mid_NPD=df_low_mid_NPD['126d FWD%'].values.tolist()
fwd_252_low_mid_NPD=df_low_mid_NPD['252d FWD%'].values.tolist()
fwd_VIX_5_low_mid_NPD=df_low_mid_NPD['5d FWD VIX'].values.tolist()
fwd_VIX_21_low_mid_NPD=df_low_mid_NPD['21d FWD VIX'].values.tolist()

###Mid NPD filtered heatmap data
seasonals_mid_NPD=df_mid_NPD['Seasonal_rank'].values.tolist()
trailing_5_mid_NPD=df_mid_NPD['5dt'].values.tolist()
trailing_21_mid_NPD=df_mid_NPD['21dt'].values.tolist()
fwd_1_mid_NPD=df_mid_NPD['1d FWD%'].values.tolist()
fwd_5_mid_NPD=df_mid_NPD['5d FWD%'].values.tolist()
fwd_10_mid_NPD=df_mid_NPD['10d FWD%'].values.tolist()
fwd_21_mid_NPD=df_mid_NPD['21d FWD%'].values.tolist()
fwd_63_mid_NPD=df_mid_NPD['63d FWD%'].values.tolist()
fwd_126_mid_NPD=df_mid_NPD['126d FWD%'].values.tolist()
fwd_252_mid_NPD=df_mid_NPD['252d FWD%'].values.tolist()
fwd_VIX_5_mid_NPD=df_mid_NPD['5d FWD VIX'].values.tolist()
fwd_VIX_21_mid_NPD=df_mid_NPD['21d FWD VIX'].values.tolist()

###High NPD filtered heatmap data
seasonals_high_NPD=df_high_NPD['Seasonal_rank'].values.tolist()
trailing_5_high_NPD=df_high_NPD['5dt'].values.tolist()
trailing_21_high_NPD=df_high_NPD['21dt'].values.tolist()
fwd_1_high_NPD=df_high_NPD['1d FWD%'].values.tolist()
fwd_5_high_NPD=df_high_NPD['5d FWD%'].values.tolist()
fwd_10_high_NPD=df_high_NPD['10d FWD%'].values.tolist()
fwd_21_high_NPD=df_high_NPD['21d FWD%'].values.tolist()
fwd_63_high_NPD=df_high_NPD['63d FWD%'].values.tolist()
fwd_126_high_NPD=df_high_NPD['126d FWD%'].values.tolist()
fwd_252_high_NPD=df_high_NPD['252d FWD%'].values.tolist()
fwd_VIX_5_high_NPD=df_high_NPD['5d FWD VIX'].values.tolist()
fwd_VIX_21_high_NPD=df_high_NPD['21d FWD VIX'].values.tolist()

print(len(fwd_5),len(fwd_5_high_NPD))
fig = make_subplots(
    rows=3, cols=2,
    specs=[[{}, {}],
           [{}, {}],
           [{"colspan": 2}, None]],
    subplot_titles=("","", "","",""))

fig.add_trace(go.Heatmap(
	z=fwd_5,
	x=npd, 
	y=trailing_21,
	colorscale=seismic,
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=1, col=1)
fig.add_trace(go.Heatmap(
	z=fwd_5,
	x=seasonals, 
	y=trailing_21,
	colorscale=seismic,
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=1,col=2)
fig.add_trace(go.Heatmap(
	z=fwd_5,
	x=npd, 
	y=trailing_5,
	colorscale=seismic,
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=2,col=1)
fig.add_trace(go.Heatmap(
	z=fwd_5,
	x=seasonals, 
	y=trailing_5,
	colorscale=seismic,
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=2,col=2)
fig.add_trace(go.Heatmap(
	z=fwd_5,
	x=npd, 
	y=seasonals,
	colorscale=seismic,
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=3,col=1)
#Update xaxis properties
fig.update_xaxes(title_text="NPD 5 Day Moving Average", range=[-15, 15],row=1, col=1)
fig.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig.update_xaxes(title_text="NPD 5 Day Moving Average", range=[-15, 15], row=2, col=1)
fig.update_xaxes(title_text="Seasonals Rank",row=2, col=2)
fig.update_xaxes(title_text="NPD 5 Day Moving Average",range=[-15, 15],row=3, col=1)

# Update yaxis properties
fig.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=1)
fig.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=2)
fig.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=2)
fig.update_yaxes(title_text="Seasonals Rank", row=3, col=1)

#Add lines pointing to todays value coordinates
fig.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=1)
fig.add_vline(x=NPD_today,line_width=3, line_color="black",row=1,col=1)
fig.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=2)
fig.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig.add_vline(x=NPD_today,line_width=3, line_color="black",row=2,col=1)
fig.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=2)
fig.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)
fig.add_hline(y=seasonal_today,line_width=3, line_color="black",row=3,col=1)
fig.add_vline(x=NPD_today,line_width=3, line_color="black",row=3,col=1)

fig.update_layout(title_text="Five Day Forward Returns")

fig2 = make_subplots(
    rows=3, cols=2,
    specs=[[{}, {}],
           [{}, {}],
           [{"colspan": 2}, None]],
    subplot_titles=("","", "","",""))

fig2.add_trace(go.Heatmap(
	z=fwd_21,
	x=npd, 
	y=trailing_21,
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=1, col=1)
fig2.add_trace(go.Heatmap(
	z=fwd_21,
	x=seasonals, 
	y=trailing_21,
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=1,col=2)
fig2.add_trace(go.Heatmap(
	z=fwd_21,
	x=npd, 
	y=trailing_5,
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=2,col=1)
fig2.add_trace(go.Heatmap(
	z=fwd_21,
	x=seasonals, 
	y=trailing_5,
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=2,col=2)
fig2.add_trace(go.Heatmap(
	z=fwd_21,
	x=npd, 
	y=seasonals,
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=3,col=1)
#Update xaxis properties
fig2.update_xaxes(title_text="NPD 5 Day Moving Average", range=[-15, 15],row=1, col=1)
fig2.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig2.update_xaxes(title_text="NPD 5 Day Moving Average", range=[-15, 15], row=2, col=1)
fig2.update_xaxes(title_text="Seasonals Rank",row=2, col=2)
fig2.update_xaxes(title_text="NPD 5 Day Moving Average",range=[-15, 15],row=3, col=1)

# Update yaxis properties
fig2.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=1)
fig2.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=2)
fig2.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig2.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=2)
fig2.update_yaxes(title_text="Seasonals Rank", row=3, col=1)

#Add lines pointing to todays value coordinates
fig2.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=1)
fig2.add_vline(x=NPD_today,line_width=3, line_color="black",row=1,col=1)
fig2.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=2)
fig2.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig2.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig2.add_vline(x=NPD_today,line_width=3, line_color="black",row=2,col=1)
fig2.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=2)
fig2.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)
fig2.add_hline(y=seasonal_today,line_width=3, line_color="black",row=3,col=1)
fig2.add_vline(x=NPD_today,line_width=3, line_color="black",row=3,col=1)

fig2.update_layout(title_text="Twenty One Day Forward Returns")

fig3 = make_subplots(
    rows=2, cols=2,
    specs=[[{}, {}],
           [{}, {}]],
    subplot_titles=("NPD <-10","-10 < NPD <-5", "-5 < NPD < 5","NPD > 5"))

fig3.add_trace(go.Heatmap(
	z=fwd_5_low_NPD,
	x=seasonals_low_NPD, 
	y=trailing_5_low_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=1, col=1)
fig3.add_trace(go.Heatmap(
	z=fwd_5_low_mid_NPD,
	x=seasonals_low_mid_NPD, 
	y=trailing_5_low_mid_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=1,col=2)
fig3.add_trace(go.Heatmap(
	z=fwd_5_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=2,col=1)
fig3.add_trace(go.Heatmap(
	z=fwd_5_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=2,col=2)
#Update xaxis properties
fig3.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig3.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig3.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig3.update_xaxes(title_text="Seasonals Rank",row=2, col=2)


# Update yaxis properties
fig3.update_yaxes(title_text="5 Day Trailing Returns", row=1, col=1)
fig3.update_yaxes(title_text="5 Day Trailing Returns", row=1, col=2)
fig3.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig3.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=2)


#Add lines pointing to todays value coordinates
fig3.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=1,col=1)
fig3.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig3.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=1,col=2)
fig3.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig3.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig3.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig3.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=2)
fig3.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)


fig3.update_layout(title_text="Five Day Forward Returns")


fig4 = make_subplots(
    rows=2, cols=2,
    specs=[[{}, {}],
           [{}, {}]],
    subplot_titles=("NPD <-10","-10 < NPD <-5", "-5 < NPD < 5","NPD > 5"))

fig4.add_trace(go.Heatmap(
	z=fwd_21_low_NPD,
	x=seasonals_low_NPD, 
	y=trailing_5_low_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-15,
	zmax=15,
	zsmooth="best"),row=1, col=1)
fig4.add_trace(go.Heatmap(
	z=fwd_21_low_mid_NPD,
	x=seasonals_low_mid_NPD, 
	y=trailing_5_low_mid_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-15,
	zmax=15,
	zsmooth="best"),row=1,col=2)
fig4.add_trace(go.Heatmap(
	z=fwd_21_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-15,
	zmax=15,
	zsmooth="best"),row=2,col=1)
fig4.add_trace(go.Heatmap(
	z=fwd_21_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-15,
	zmax=15,
	zsmooth="best"),row=2,col=2)
#Update xaxis properties
fig4.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig4.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig4.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig4.update_xaxes(title_text="Seasonals Rank",row=2, col=2)


# Update yaxis properties
fig4.update_yaxes(title_text="5 Day Trailing Returns", row=1, col=1)
fig4.update_yaxes(title_text="5 Day Trailing Returns", row=1, col=2)
fig4.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig4.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=2)


#Add lines pointing to todays value coordinates
fig4.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=1,col=1)
fig4.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig4.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=1,col=2)
fig4.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig4.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig4.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig4.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=2)
fig4.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)


fig4.update_layout(title_text="Twenty-One Day Forward Returns")


fig5 = make_subplots(
    rows=3, cols=2,
    specs=[[{}, {}],
           [{}, {}],
           [{}, {}]],
    subplot_titles=("1d FWD","5d FWD","21d FWD","63d FWD","126d FWD","252d FWD"))

fig5.add_trace(go.Heatmap(
	z=fwd_1_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.88),
	colorscale=seismic,
	reversescale=True,
	zmin=-3,
	zmax=3,
	zsmooth="best"),row=1, col=1)
fig5.add_trace(go.Heatmap(
	z=fwd_5_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorbar=dict(len=0.2, y=0.88),
	colorscale=seismic,
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=1,col=2)
fig5.add_trace(go.Heatmap(
	z=fwd_21_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.5),
	colorscale=seismic,
	reversescale=True,
	zmin=-15,
	zmax=15,
	zsmooth="best"),row=2,col=1)
fig5.add_trace(go.Heatmap(
	z=fwd_63_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorbar=dict(len=0.2, y=0.5),
	colorscale=seismic,
	reversescale=True,
	zmin=-20,
	zmax=20,
	zsmooth="best"),row=2,col=2)
fig5.add_trace(go.Heatmap(
	z=fwd_126_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.12),
	colorscale=seismic,
	reversescale=True,
	zmin=-20,
	zmax=20,
	zsmooth="best"),row=3,col=1)
fig5.add_trace(go.Heatmap(
	z=fwd_252_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorbar=dict(len=0.2, y=0.12),
	colorscale=seismic,
	reversescale=True,
	zmin=-25,
	zmax=25,
	zsmooth="best"),row=3,col=2)
#Update xaxis properties
fig5.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig5.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig5.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig5.update_xaxes(title_text="Seasonals Rank",row=2, col=2)
fig5.update_xaxes(title_text="Seasonals Rank", row=3, col=1)
fig5.update_xaxes(title_text="Seasonals Rank",row=3, col=2)

# Update yaxis properties
fig5.update_yaxes(title_text="5 Day Trailing Returns", row=1, col=1)
fig5.update_yaxes(title_text="5 Day Trailing Returns", row=1, col=2)
fig5.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig5.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=2)
fig5.update_yaxes(title_text="5 Day Trailing Returns", row=3, col=1)
fig5.update_yaxes(title_text="5 Day Trailing Returns", row=3, col=2)

#Add lines pointing to todays value coordinates
fig5.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=1,col=1)
fig5.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig5.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=1,col=2)
fig5.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig5.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig5.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig5.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=2)
fig5.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)
fig5.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=3,col=1)
fig5.add_vline(x=seasonal_today,line_width=3, line_color="black",row=3,col=1)
fig5.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=3,col=2)
fig5.add_vline(x=seasonal_today,line_width=3, line_color="black",row=3,col=2)

fig5.update_layout(title_text="Forward looking returns when NPD is >5")

fig6 = make_subplots(
    rows=2, cols=2,
    specs=[[{}, {}],
           [{}, {}]],
    subplot_titles=("NPD <-10","-10 < NPD <-5", "-5 < NPD < 5","NPD > 5"))

fig6.add_trace(go.Heatmap(
	z=fwd_5_low_NPD,
	x=seasonals_low_NPD, 
	y=trailing_21_low_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=1, col=1)
fig6.add_trace(go.Heatmap(
	z=fwd_5_low_mid_NPD,
	x=seasonals_low_mid_NPD, 
	y=trailing_21_low_mid_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=1,col=2)
fig6.add_trace(go.Heatmap(
	z=fwd_5_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_21_mid_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=2,col=1)
fig6.add_trace(go.Heatmap(
	z=fwd_5_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_21_high_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=2,col=2)
#Update xaxis properties
fig6.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig6.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig6.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig6.update_xaxes(title_text="Seasonals Rank",row=2, col=2)


# Update yaxis properties
fig6.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=1)
fig6.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=2)
fig6.update_yaxes(title_text="21 Day Trailing Returns", row=2, col=1)
fig6.update_yaxes(title_text="21 Day Trailing Returns", row=2, col=2)


#Add lines pointing to todays value coordinates
fig6.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=1)
fig6.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig6.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=2)
fig6.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig6.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=2,col=1)
fig6.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig6.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=2,col=2)
fig6.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)


fig6.update_layout(title_text="Five Day Forward Returns")


fig7 = make_subplots(
    rows=2, cols=2,
    specs=[[{}, {}],
           [{}, {}]],
    subplot_titles=("NPD <-10","-10 < NPD <-5", "-5 < NPD < 5","NPD > 5"))

fig7.add_trace(go.Heatmap(
	z=fwd_21_low_NPD,
	x=seasonals_low_NPD, 
	y=trailing_21_low_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-15,
	zmax=15,
	zsmooth="best"),row=1, col=1)
fig7.add_trace(go.Heatmap(
	z=fwd_21_low_mid_NPD,
	x=seasonals_low_mid_NPD, 
	y=trailing_21_low_mid_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-15,
	zmax=15,
	zsmooth="best"),row=1,col=2)
fig7.add_trace(go.Heatmap(
	z=fwd_21_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_21_mid_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-15,
	zmax=15,
	zsmooth="best"),row=2,col=1)
fig7.add_trace(go.Heatmap(
	z=fwd_21_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_21_high_NPD,
	colorscale='Jet',
	reversescale=True,
	zmin=-15,
	zmax=15,
	zsmooth="best"),row=2,col=2)
#Update xaxis properties
fig7.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig7.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig7.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig7.update_xaxes(title_text="Seasonals Rank",row=2, col=2)


# Update yaxis properties
fig7.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=1)
fig7.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=2)
fig7.update_yaxes(title_text="21 Day Trailing Returns", row=2, col=1)
fig7.update_yaxes(title_text="21 Day Trailing Returns", row=2, col=2)


#Add lines pointing to todays value coordinates
fig7.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=1)
fig7.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig7.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=2)
fig7.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig7.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=2,col=1)
fig7.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig7.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=2,col=2)
fig7.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)

fig7.update_layout(title_text="Twenty-One Day Forward Returns")

fig8 = make_subplots(
    rows=2, cols=2,
    vertical_spacing=0.05,
    specs=[[{"b":0.1}, {"b":0.1}],
           [{"b":0.1}, {"b":0.1}]],
    subplot_titles=("21d Fwd Returns","5d Fwd Returns",
    	"21d Fwd Returns","5d Fwd Returns"))

fig8.add_trace(go.Heatmap(
	z=fwd_21_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_21_high_NPD,
	colorbar=dict(len=0.36, x=0.47,y=0.82),
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=1, col=1)
fig8.add_trace(go.Heatmap(
	z=fwd_5_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_21_high_NPD,
	colorbar=dict(len=0.36,y=0.82),
	colorscale=seismic,
	reversescale=True,
	zmin=-5,
	zmax=5,
	zsmooth="best"),row=1,col=2)
fig8.add_trace(go.Heatmap(
	z=fwd_21_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorbar=dict(len=0.36, x=0.47,y=0.28),
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=2,col=1)
fig8.add_trace(go.Heatmap(
	z=fwd_5_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorbar=dict(len=0.36,y=0.28),
	colorscale=seismic,
	reversescale=True,
	zmin=-5,
	zmax=5,
	zsmooth="best"),row=2,col=2)

#Update xaxis properties
fig8.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig8.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig8.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig8.update_xaxes(title_text="Seasonals Rank",row=2, col=2)

# Update yaxis properties
fig8.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=1)
fig8.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=2)
fig8.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig8.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=2)

#Add lines pointing to todays value coordinates
fig8.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=1)
fig8.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig8.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=2)
fig8.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig8.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig8.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig8.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=2)
fig8.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)


fig8.add_annotation(
    showarrow=False,
    text=note,
    font=dict(size=12), 
    xref='paper',
    x=0,
    yref='paper',
    y=-0.05,
    xshift=0,
    yshift=0,
    xanchor='left',
    yanchor='bottom',
    align='left',
    )
fig8.update_layout(title_text=f'NPD 5 Day Average > 50th Percentile')
fig8.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_21_up1-.2,x1=seasonal_tm+.15,y1=rnk_21_up1+.2), row=1, col="all", line_color="green"
)
fig8.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_21_down1-.2,x1=seasonal_tm+.15,y1=rnk_21_down1+.2), row=1, col="all", line_color="red"
)
fig8.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_up1-.2,x1=seasonal_tm+.15,y1=rnk_5_up1+.2), row=2, col="all", line_color="green"
)
fig8.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_down1-.2,x1=seasonal_tm+.15,y1=rnk_5_down1+.2), row=2, col="all", line_color="red"
)
fig8.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_flat-.2,x1=seasonal_tm+.15,y1=rnk_5_flat+.2), row=2, col="all", line_color="grey"
)
fig8.add_shape(
	dict(type="circle",x0=seasonal_fwd_5-.15,y0=trailing_5_today-.2,x1=seasonal_fwd_5+.15,y1=trailing_5_today+.2), row=2, col="all", line_color="gold"
)
fig8.add_shape(
	dict(type="circle",x0=seasonal_fwd_5-.15,y0=trailing_21_today-.2,x1=seasonal_fwd_5+.15,y1=trailing_21_today+.2), row=1, col="all", line_color="gold"
)


fig9 = make_subplots(
    rows=3, cols=2,
    specs=[[{}, {}],
           [{}, {}],
           [{}, {}]],
    subplot_titles=("1d FWD","1d FWD","5d FWD","5d FWD","21d FWD","21d FWD"))

fig9.add_trace(go.Heatmap(
	z=fwd_1_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.88),
	colorscale=seismic,
	reversescale=True,
	zmin=-3,
	zmax=3,
	zsmooth="best"),row=1, col=1)
fig9.add_trace(go.Heatmap(
	z=fwd_1_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_21_high_NPD,
	colorbar=dict(len=0.2, y=0.88),
	colorscale=seismic,
	reversescale=True,
	zmin=-3,
	zmax=3,
	zsmooth="best"),row=1,col=2)
fig9.add_trace(go.Heatmap(
	z=fwd_5_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.5),
	colorscale=seismic,
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=2,col=1)
fig9.add_trace(go.Heatmap(
	z=fwd_5_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_21_high_NPD,
	colorbar=dict(len=0.2, y=0.5),
	colorscale=seismic,
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=2,col=2)
fig9.add_trace(go.Heatmap(
	z=fwd_21_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.12),
	colorscale=seismic,
	reversescale=True,
	zmin=-15,
	zmax=15,
	zsmooth="best"),row=3,col=1)
fig9.add_trace(go.Heatmap(
	z=fwd_21_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_21_high_NPD,
	colorbar=dict(len=0.2, y=0.12),
	colorscale=seismic,
	reversescale=True,
	zmin=-15,
	zmax=15,
	zsmooth="best"),row=3,col=2)
#Update xaxis properties
fig9.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig9.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig9.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig9.update_xaxes(title_text="Seasonals Rank",row=2, col=2)
fig9.update_xaxes(title_text="Seasonals Rank", row=3, col=1)
fig9.update_xaxes(title_text="Seasonals Rank",row=3, col=2)

# Update yaxis properties
fig9.update_yaxes(title_text="5 Day Trailing Returns", row=1, col=1)
fig9.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=2)
fig9.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig9.update_yaxes(title_text="21 Day Trailing Returns", row=2, col=2)
fig9.update_yaxes(title_text="5 Day Trailing Returns", row=3, col=1)
fig9.update_yaxes(title_text="21 Day Trailing Returns", row=3, col=2)

#Add lines pointing to todays value coordinates
fig9.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=1,col=1)
fig9.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig9.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=2)
fig9.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig9.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig9.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig9.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=2,col=2)
fig9.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)
fig9.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=3,col=1)
fig9.add_vline(x=seasonal_today,line_width=3, line_color="black",row=3,col=1)
fig9.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=3,col=2)
fig9.add_vline(x=seasonal_today,line_width=3, line_color="black",row=3,col=2)

fig9.update_layout(title_text="Forward looking returns when NPD is >5")

fig10 = make_subplots(
    rows=3, cols=2,
    specs=[[{}, {}],
           [{}, {}],
           [{}, {}]],
    subplot_titles=("1d FWD","1d FWD","5d FWD","5d FWD","21d FWD","21d FWD"))

fig10.add_trace(go.Heatmap(
	z=fwd_1_low_mid_NPD,
	x=seasonals_low_mid_NPD, 
	y=trailing_5_low_mid_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.88),
	colorscale=seismic,
	reversescale=True,
	zmin=-3,
	zmax=3,
	zsmooth="best"),row=1, col=1)
fig10.add_trace(go.Heatmap(
	z=fwd_1_low_mid_NPD,
	x=seasonals_low_mid_NPD, 
	y=trailing_21_low_mid_NPD,
	colorbar=dict(len=0.2, y=0.88),
	colorscale=seismic,
	reversescale=True,
	zmin=-3,
	zmax=3,
	zsmooth="best"),row=1,col=2)
fig10.add_trace(go.Heatmap(
	z=fwd_5_low_mid_NPD,
	x=seasonals_low_mid_NPD, 
	y=trailing_5_low_mid_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.5),
	colorscale=seismic,
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=2,col=1)
fig10.add_trace(go.Heatmap(
	z=fwd_5_low_mid_NPD,
	x=seasonals_low_mid_NPD, 
	y=trailing_21_low_mid_NPD,
	colorbar=dict(len=0.2, y=0.5),
	colorscale=seismic,
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=2,col=2)
fig10.add_trace(go.Heatmap(
	z=fwd_21_low_mid_NPD,
	x=seasonals_low_mid_NPD, 
	y=trailing_5_low_mid_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.12),
	colorscale=seismic,
	reversescale=True,
	zmin=-15,
	zmax=15,
	zsmooth="best"),row=3,col=1)
fig10.add_trace(go.Heatmap(
	z=fwd_21_low_mid_NPD,
	x=seasonals_low_mid_NPD, 
	y=trailing_21_low_mid_NPD,
	colorbar=dict(len=0.2, y=0.12),
	colorscale=seismic,
	reversescale=True,
	zmin=-15,
	zmax=15,
	zsmooth="best"),row=3,col=2)
#Update xaxis properties
fig10.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig10.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig10.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig10.update_xaxes(title_text="Seasonals Rank",row=2, col=2)
fig10.update_xaxes(title_text="Seasonals Rank", row=3, col=1)
fig10.update_xaxes(title_text="Seasonals Rank",row=3, col=2)

# Update yaxis properties
fig10.update_yaxes(title_text="5 Day Trailing Returns", row=1, col=1)
fig10.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=2)
fig10.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig10.update_yaxes(title_text="21 Day Trailing Returns", row=2, col=2)
fig10.update_yaxes(title_text="5 Day Trailing Returns", row=3, col=1)
fig10.update_yaxes(title_text="21 Day Trailing Returns", row=3, col=2)

#Add lines pointing to todays value coordinates
fig10.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=1,col=1)
fig10.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig10.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=2)
fig10.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig10.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig10.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig10.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=2,col=2)
fig10.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)
fig10.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=3,col=1)
fig10.add_vline(x=seasonal_today,line_width=3, line_color="black",row=3,col=1)
fig10.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=3,col=2)
fig10.add_vline(x=seasonal_today,line_width=3, line_color="black",row=3,col=2)

fig10.update_layout(title_text="Forward looking returns when NPD is >-10 & <-5")

fig11 = make_subplots(
    rows=2, cols=2,
    vertical_spacing=0.05,
    specs=[[{"b":0.1}, {"b":0.1}],
           [{"b":0.1}, {"b":0.1}]],
    subplot_titles=("21d Fwd Returns","5d Fwd Returns",
    	"21d Fwd Returns","5d Fwd Returns"))

fig11.add_trace(go.Heatmap(
	z=fwd_21_low_NPD,
	x=seasonals_low_NPD, 
	y=trailing_21_low_NPD,
	colorbar=dict(len=0.36, x=0.47,y=0.82),
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=1, col=1)
fig11.add_trace(go.Heatmap(
	z=fwd_5_low_NPD,
	x=seasonals_low_NPD, 
	y=trailing_21_low_NPD,
	colorbar=dict(len=0.36,y=0.82),
	colorscale=seismic,
	reversescale=True,
	zmin=-5,
	zmax=5,
	zsmooth="best"),row=1,col=2)
fig11.add_trace(go.Heatmap(
	z=fwd_21_low_NPD,
	x=seasonals_low_NPD, 
	y=trailing_5_low_NPD,
	colorbar=dict(len=0.36, x=0.47,y=0.28),
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=2,col=1)
fig11.add_trace(go.Heatmap(
	z=fwd_5_low_NPD,
	x=seasonals_low_NPD, 
	y=trailing_5_low_NPD,
	colorbar=dict(len=0.36,y=0.28),
	colorscale=seismic,
	reversescale=True,
	zmin=-5,
	zmax=5,
	zsmooth="best"),row=2,col=2)

#Update xaxis properties
fig11.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig11.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig11.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig11.update_xaxes(title_text="Seasonals Rank",row=2, col=2)

# Update yaxis properties
fig11.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=1)
fig11.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=2)
fig11.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig11.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=2)

#Add lines pointing to todays value coordinates
fig11.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=1)
fig11.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig11.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=2)
fig11.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig11.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig11.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig11.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=2)
fig11.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)
fig11.update_layout(title_text="NPD 5 Day Average < 50th Percentile")
# fig11.add_annotation(
#     showarrow=False,
#     text=note,
#     font=dict(size=12), 
#     xref='paper',
#     x=0,
#     yref='paper',
#     y=-0.05,
#     xshift=0,
#     yshift=0,
#     xanchor='left',
#     yanchor='bottom',
#     align='left',
#     )
fig11.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_21_up1-.2,x1=seasonal_tm+.15,y1=rnk_21_up1+.2), row=1, col="all", line_color="green"
)
fig11.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_21_down1-.2,x1=seasonal_tm+.15,y1=rnk_21_down1+.2), row=1, col="all", line_color="red"
)
fig11.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_up1-.2,x1=seasonal_tm+.15,y1=rnk_5_up1+.2), row=2, col="all", line_color="green"
)
fig11.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_down1-.2,x1=seasonal_tm+.15,y1=rnk_5_down1+.2), row=2, col="all", line_color="red"
)
fig11.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_flat-.2,x1=seasonal_tm+.15,y1=rnk_5_flat+.2), row=2, col="all", line_color="grey"
)
fig11.add_shape(
	dict(type="circle",x0=seasonal_fwd_5-.15,y0=trailing_5_today-.2,x1=seasonal_fwd_5+.15,y1=trailing_5_today+.2), row=2, col="all", line_color="gold"
)
fig11.add_shape(
	dict(type="circle",x0=seasonal_fwd_5-.15,y0=trailing_21_today-.2,x1=seasonal_fwd_5+.15,y1=trailing_21_today+.2), row=1, col="all", line_color="gold"
)
fig11.add_annotation(
    showarrow=False,
    text=note,
    font=dict(size=12), 
    xref='paper',
    x=0,
    yref='paper',
    y=-0.05,
    xshift=0,
    yshift=0,
    xanchor='left',
    yanchor='bottom',
    align='left',
    )

fig12 = make_subplots(
    rows=2, cols=2,
    vertical_spacing=0.05,
    specs=[[{"b":0.1}, {"b":0.1}],
           [{"b":0.1}, {"b":0.1}]],
    subplot_titles=("21d Fwd Returns","5d Fwd Returns",
    	"21d Fwd Returns","5d Fwd Returns"))

fig12.add_trace(go.Heatmap(
	z=fwd_21_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_21_mid_NPD,
	colorbar=dict(len=0.36, x=0.47,y=0.82),
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=1, col=1)
fig12.add_trace(go.Heatmap(
	z=fwd_5_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_21_mid_NPD,
	colorbar=dict(len=0.36,y=0.82),
	colorscale=seismic,
	reversescale=True,
	zmin=-5,
	zmax=5,
	zsmooth="best"),row=1,col=2)
fig12.add_trace(go.Heatmap(
	z=fwd_21_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorbar=dict(len=0.36, x=0.47,y=0.28),
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=2,col=1)
fig12.add_trace(go.Heatmap(
	z=fwd_5_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorbar=dict(len=0.36,y=0.28),
	colorscale=seismic,
	reversescale=True,
	zmin=-5,
	zmax=5,
	zsmooth="best"),row=2,col=2)

#Update xaxis properties
fig12.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig12.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig12.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig12.update_xaxes(title_text="Seasonals Rank",row=2, col=2)

# Update yaxis properties
fig12.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=1)
fig12.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=2)
fig12.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig12.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=2)

#Add lines pointing to todays value coordinates
fig12.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=1)
fig12.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig12.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=2)
fig12.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig12.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig12.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig12.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=2)
fig12.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)


fig12.add_annotation(
    showarrow=False,
    text=note,
    font=dict(size=12), 
    xref='paper',
    x=0,
    yref='paper',
    y=-0.05,
    xshift=0,
    yshift=0,
    xanchor='left',
    yanchor='bottom',
    align='left',
    )
fig12.update_layout(title_text="NPD 5 Day Average >-5 & <5")
fig12.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_21_up1-.2,x1=seasonal_tm+.15,y1=rnk_21_up1+.2), row=1, col="all", line_color="green"
)
fig12.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_21_down1-.2,x1=seasonal_tm+.15,y1=rnk_21_down1+.2), row=1, col="all", line_color="red"
)
fig12.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_up1-.2,x1=seasonal_tm+.15,y1=rnk_5_up1+.2), row=2, col="all", line_color="green"
)
fig12.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_down1-.2,x1=seasonal_tm+.15,y1=rnk_5_down1+.2), row=2, col="all", line_color="red"
)
fig12.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_flat-.2,x1=seasonal_tm+.15,y1=rnk_5_flat+.2), row=2, col="all", line_color="grey"
)
fig12.add_shape(
	dict(type="circle",x0=seasonal_fwd_5-.15,y0=trailing_5_today-.2,x1=seasonal_fwd_5+.15,y1=trailing_5_today+.2), row=2, col="all", line_color="gold"
)
fig12.add_shape(
	dict(type="circle",x0=seasonal_fwd_5-.15,y0=trailing_21_today-.2,x1=seasonal_fwd_5+.15,y1=trailing_21_today+.2), row=1, col="all", line_color="gold"
)


fig13 = make_subplots(
    rows=2, cols=2,
    specs=[[{}, {}],
           [{}, {}]],
    subplot_titles=("5d FWD VIX","21d FWD VIX",
    	"5d FWD VIX","21d FWD VIX"))

fig13.add_trace(go.Heatmap(
	z=fwd_VIX_5_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_21_high_NPD,
	colorbar=dict(len=0.36, x=0.47,y=0.82),
	colorscale=seismic,
	reversescale=False,
	zmin=10,
	zmax=40,
	zsmooth="best"),row=1, col=1)
fig13.add_trace(go.Heatmap(
	z=fwd_VIX_21_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_21_high_NPD,
	colorbar=dict(len=0.36,y=0.82),
	colorscale=seismic,
	reversescale=False,
	zmin=10,
	zmax=40,
	zsmooth="best"),row=1,col=2)
fig13.add_trace(go.Heatmap(
	z=fwd_VIX_5_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorbar=dict(len=0.36, x=0.47,y=0.2),
	colorscale=seismic,
	reversescale=False,
	zmin=10,
	zmax=40,
	zsmooth="best"),row=2,col=1)
fig13.add_trace(go.Heatmap(
	z=fwd_VIX_21_high_NPD,
	x=seasonals_high_NPD, 
	y=trailing_5_high_NPD,
	colorbar=dict(len=0.36,y=0.2),
	colorscale=seismic,
	reversescale=False,
	zmin=15,
	zmax=40,
	zsmooth="best"),row=2,col=2)

#Update xaxis properties
fig13.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig13.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig13.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig13.update_xaxes(title_text="Seasonals Rank",row=2, col=2)

# Update yaxis properties
fig13.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=1)
fig13.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=2)
fig13.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig13.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=2)

#Add lines pointing to todays value coordinates
fig13.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=1)
fig13.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig13.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=2)
fig13.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig13.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig13.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig13.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=2)
fig13.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)


fig13.update_layout(title_text="FWD VIX When NPD 5 Day Average > 5")

fig14 = make_subplots(
    rows=2, cols=2,
    vertical_spacing=0.05,
    specs=[[{"b":0.1}, {"b":0.1}],
           [{"b":0.1}, {"b":0.1}]],
    subplot_titles=("21d Fwd Returns","5d Fwd Returns",
    	"21d Fwd Returns","5d Fwd Returns"))

fig14.add_trace(go.Heatmap(
	z=fwd_21_low_mid_NPD,
	x=seasonals_low_mid_NPD, 
	y=trailing_21_low_mid_NPD,
	colorbar=dict(len=0.36, x=0.47,y=0.82),
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=1, col=1)
fig14.add_trace(go.Heatmap(
	z=fwd_5_low_mid_NPD,
	x=seasonals_low_mid_NPD, 
	y=trailing_21_low_mid_NPD,
	colorbar=dict(len=0.36,y=0.82),
	colorscale=seismic,
	reversescale=True,
	zmin=-5,
	zmax=5,
	zsmooth="best"),row=1,col=2)
fig14.add_trace(go.Heatmap(
	z=fwd_21_low_mid_NPD,
	x=seasonals_low_mid_NPD, 
	y=trailing_5_low_mid_NPD,
	colorbar=dict(len=0.36, x=0.47,y=0.28),
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=2,col=1)
fig14.add_trace(go.Heatmap(
	z=fwd_5_low_mid_NPD,
	x=seasonals_low_mid_NPD, 
	y=trailing_5_low_mid_NPD,
	colorbar=dict(len=0.36,y=0.28),
	colorscale=seismic,
	reversescale=True,
	zmin=-5,
	zmax=5,
	zsmooth="best"),row=2,col=2)

#Update xaxis properties
fig14.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig14.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig14.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig14.update_xaxes(title_text="Seasonals Rank",row=2, col=2)

# Update yaxis properties
fig14.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=1)
fig14.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=2)
fig14.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig14.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=2)

#Add lines pointing to todays value coordinates
fig14.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=1)
fig14.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig14.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=2)
fig14.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig14.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig14.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig14.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=2)
fig14.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)


fig14.add_annotation(
    showarrow=False,
    text=note,
    font=dict(size=12), 
    xref='paper',
    x=0,
    yref='paper',
    y=-0.05,
    xshift=0,
    yshift=0,
    xanchor='left',
    yanchor='bottom',
    align='left',
    )
fig14.update_layout(title_text="NPD 5 Day Average >-10 & <-5")
fig14.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_21_up1-.2,x1=seasonal_tm+.15,y1=rnk_21_up1+.2), row=1, col="all", line_color="green"
)
fig14.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_21_down1-.2,x1=seasonal_tm+.15,y1=rnk_21_down1+.2), row=1, col="all", line_color="red"
)
fig14.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_up1-.2,x1=seasonal_tm+.15,y1=rnk_5_up1+.2), row=2, col="all", line_color="green"
)
fig14.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_down1-.2,x1=seasonal_tm+.15,y1=rnk_5_down1+.2), row=2, col="all", line_color="red"
)
fig14.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_flat-.2,x1=seasonal_tm+.15,y1=rnk_5_flat+.2), row=2, col="all", line_color="grey"
)
fig14.add_shape(
	dict(type="circle",x0=seasonal_fwd_5-.15,y0=trailing_5_today-.2,x1=seasonal_fwd_5+.15,y1=trailing_5_today+.2), row=2, col="all", line_color="gold"
)
fig14.add_shape(
	dict(type="circle",x0=seasonal_fwd_5-.15,y0=trailing_21_today-.2,x1=seasonal_fwd_5+.15,y1=trailing_21_today+.2), row=1, col="all", line_color="gold"
)

fig15 = make_subplots(
    rows=2, cols=2,
    vertical_spacing=0.05,
    specs=[[{"b":0.1}, {"b":0.1}],
           [{"b":0.1}, {"b":0.1}]],
    subplot_titles=("21d Fwd VIX","5d Fwd VIX",
    	"21d Fwd VIX","5d Fwd VIX"))

fig15.add_trace(go.Heatmap(
	z=fwd_VIX_21_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_21_mid_NPD,
	colorbar=dict(len=0.36, x=0.47,y=0.82),
	colorscale=seismic,
	reversescale=False,
	zmin=10,
	zmax=40,
	zsmooth="best"),row=1, col=1)
fig15.add_trace(go.Heatmap(
	z=fwd_VIX_5_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_21_mid_NPD,
	colorbar=dict(len=0.36,y=0.82),
	colorscale=seismic,
	reversescale=False,
	zmin=10,
	zmax=40,
	zsmooth="best"),row=1,col=2)
fig15.add_trace(go.Heatmap(
	z=fwd_VIX_21_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorbar=dict(len=0.36, x=0.47,y=0.28),
	colorscale=seismic,
	reversescale=False,
	zmin=10,
	zmax=40,
	zsmooth="best"),row=2,col=1)
fig15.add_trace(go.Heatmap(
	z=fwd_VIX_5_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorbar=dict(len=0.36,y=0.28),
	colorscale=seismic,
	reversescale=False,
	zmin=10,
	zmax=40,
	zsmooth="best"),row=2,col=2)

#Update xaxis properties
fig15.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig15.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig15.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig15.update_xaxes(title_text="Seasonals Rank",row=2, col=2)

# Update yaxis properties
fig15.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=1)
fig15.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=2)
fig15.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig15.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=2)

#Add lines pointing to todays value coordinates
fig15.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=1)
fig15.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig15.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=2)
fig15.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig15.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig15.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig15.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=2)
fig15.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)


fig15.add_annotation(
    showarrow=False,
    text=note,
    font=dict(size=12), 
    xref='paper',
    x=0,
    yref='paper',
    y=-0.05,
    xshift=0,
    yshift=0,
    xanchor='left',
    yanchor='bottom',
    align='left',
    )
fig15.update_layout(title_text="NPD 5 Day Average >-5 & <5")
fig15.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_21_up1-.2,x1=seasonal_tm+.15,y1=rnk_21_up1+.2), row=1, col="all", line_color="green"
)
fig15.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_21_down1-.2,x1=seasonal_tm+.15,y1=rnk_21_down1+.2), row=1, col="all", line_color="red"
)
fig15.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_up1-.2,x1=seasonal_tm+.15,y1=rnk_5_up1+.2), row=2, col="all", line_color="green"
)
fig15.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_down1-.2,x1=seasonal_tm+.15,y1=rnk_5_down1+.2), row=2, col="all", line_color="red"
)
fig15.add_shape(
	dict(type="circle",x0=seasonal_tm-.15,y0=rnk_5_flat-.2,x1=seasonal_tm+.15,y1=rnk_5_flat+.2), row=2, col="all", line_color="grey"
)
fig15.add_shape(
	dict(type="circle",x0=seasonal_fwd_5-.15,y0=trailing_5_today-.2,x1=seasonal_fwd_5+.15,y1=trailing_5_today+.2), row=2, col="all", line_color="gold"
)
fig15.add_shape(
	dict(type="circle",x0=seasonal_fwd_5-.15,y0=trailing_21_today-.2,x1=seasonal_fwd_5+.15,y1=trailing_21_today+.2), row=1, col="all", line_color="gold"
)

fig16 = make_subplots(
    rows=3, cols=2,
    specs=[[{}, {}],
           [{}, {}],
           [{}, {}]],
    subplot_titles=("1d FWD","1d FWD","5d FWD","5d FWD","21d FWD","21d FWD"))

fig16.add_trace(go.Heatmap(
	z=fwd_1_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.88),
	colorscale=seismic,
	reversescale=True,
	zmin=-3,
	zmax=3,
	zsmooth="best"),row=1, col=1)
fig16.add_trace(go.Heatmap(
	z=fwd_5_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorbar=dict(len=0.2, y=0.88),
	colorscale=seismic,
	reversescale=True,
	zmin=-7,
	zmax=7,
	zsmooth="best"),row=1,col=2)
fig16.add_trace(go.Heatmap(
	z=fwd_21_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.5),
	colorscale=seismic,
	reversescale=True,
	zmin=-15,
	zmax=15,
	zsmooth="best"),row=2,col=1)
fig16.add_trace(go.Heatmap(
	z=fwd_63_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorbar=dict(len=0.2, y=0.5),
	colorscale=seismic,
	reversescale=True,
	zmin=-20,
	zmax=20,
	zsmooth="best"),row=2,col=2)
fig16.add_trace(go.Heatmap(
	z=fwd_126_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.12),
	colorscale=seismic,
	reversescale=True,
	zmin=-20,
	zmax=20,
	zsmooth="best"),row=3,col=1)
fig16.add_trace(go.Heatmap(
	z=fwd_252_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorbar=dict(len=0.2, y=0.12),
	colorscale=seismic,
	reversescale=True,
	zmin=-25,
	zmax=25,
	zsmooth="best"),row=3,col=2)
#Update xaxis properties
fig16.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig16.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig16.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig16.update_xaxes(title_text="Seasonals Rank",row=2, col=2)
fig16.update_xaxes(title_text="Seasonals Rank", row=3, col=1)
fig16.update_xaxes(title_text="Seasonals Rank",row=3, col=2)

# Update yaxis properties
fig16.update_yaxes(title_text="5 Day Trailing Returns", row=1, col=1)
fig16.update_yaxes(title_text="5 Day Trailing Returns", row=1, col=2)
fig16.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig16.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=2)
fig16.update_yaxes(title_text="5 Day Trailing Returns", row=3, col=1)
fig16.update_yaxes(title_text="5 Day Trailing Returns", row=3, col=2)

#Add lines pointing to todays value coordinates
fig16.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=1,col=1)
fig16.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig16.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=1,col=2)
fig16.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig16.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig16.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig16.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=2)
fig16.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)
fig16.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=3,col=1)
fig16.add_vline(x=seasonal_today,line_width=3, line_color="black",row=3,col=1)
fig16.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=3,col=2)
fig16.add_vline(x=seasonal_today,line_width=3, line_color="black",row=3,col=2)

fig16.update_layout(title_text="Forward looking returns when NPD is >-5 and <5")

fig17 = make_subplots(
    rows=3, cols=2,
    specs=[[{}, {}],
           [{}, {}],
           [{}, {}]],
    subplot_titles=("1d FWD","1d FWD","5d FWD","5d FWD","21d FWD","21d FWD"))

fig17.add_trace(go.Heatmap(
	z=fwd_1_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.88),
	colorscale=seismic,
	reversescale=True,
	zmin=-2,
	zmax=2,
	zsmooth="best"),row=1, col=1)
fig17.add_trace(go.Heatmap(
	z=fwd_1_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_21_mid_NPD,
	colorbar=dict(len=0.2, y=0.88),
	colorscale=seismic,
	reversescale=True,
	zmin=-2,
	zmax=2,
	zsmooth="best"),row=1,col=2)
fig17.add_trace(go.Heatmap(
	z=fwd_5_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.5),
	colorscale=seismic,
	reversescale=True,
	zmin=-5,
	zmax=5,
	zsmooth="best"),row=2,col=1)
fig17.add_trace(go.Heatmap(
	z=fwd_5_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_21_mid_NPD,
	colorbar=dict(len=0.2, y=0.5),
	colorscale=seismic,
	reversescale=True,
	zmin=-5,
	zmax=5,
	zsmooth="best"),row=2,col=2)
fig17.add_trace(go.Heatmap(
	z=fwd_21_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_5_mid_NPD,
	colorbar=dict(len=0.2, x=0.47, y=0.12),
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=3,col=1)
fig17.add_trace(go.Heatmap(
	z=fwd_21_mid_NPD,
	x=seasonals_mid_NPD, 
	y=trailing_21_mid_NPD,
	colorbar=dict(len=0.2, y=0.12),
	colorscale=seismic,
	reversescale=True,
	zmin=-10,
	zmax=10,
	zsmooth="best"),row=3,col=2)
#Update xaxis properties
fig17.update_xaxes(title_text="Seasonals Rank",row=1, col=1)
fig17.update_xaxes(title_text="Seasonals Rank", row=1, col=2)
fig17.update_xaxes(title_text="Seasonals Rank", row=2, col=1)
fig17.update_xaxes(title_text="Seasonals Rank",row=2, col=2)
fig17.update_xaxes(title_text="Seasonals Rank", row=3, col=1)
fig17.update_xaxes(title_text="Seasonals Rank",row=3, col=2)

# Update yaxis properties
fig17.update_yaxes(title_text="5 Day Trailing Returns", row=1, col=1)
fig17.update_yaxes(title_text="21 Day Trailing Returns", row=1, col=2)
fig17.update_yaxes(title_text="5 Day Trailing Returns", row=2, col=1)
fig17.update_yaxes(title_text="21 Day Trailing Returns", row=2, col=2)
fig17.update_yaxes(title_text="5 Day Trailing Returns", row=3, col=1)
fig17.update_yaxes(title_text="21 Day Trailing Returns", row=3, col=2)

#Add lines pointing to todays value coordinates
fig17.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=1,col=1)
fig17.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=1)
fig17.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=1,col=2)
fig17.add_vline(x=seasonal_today,line_width=3, line_color="black",row=1,col=2)
fig17.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=2,col=1)
fig17.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=1)
fig17.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=2,col=2)
fig17.add_vline(x=seasonal_today,line_width=3, line_color="black",row=2,col=2)
fig17.add_hline(y=trailing_5_today,line_width=3, line_color="black",row=3,col=1)
fig17.add_vline(x=seasonal_today,line_width=3, line_color="black",row=3,col=1)
fig17.add_hline(y=trailing_21_today,line_width=3, line_color="black",row=3,col=2)
fig17.add_vline(x=seasonal_today,line_width=3, line_color="black",row=3,col=2)

fig17.update_layout(title_text="Forward looking returns when NPD is >-5 and <5")

##figs explained....
##Fig= 5 plots that look @ 5d fwds given mix of different predictors
##fig2= same as fig 1 but looks @ 21d fwds rather than 5d 
##fig3= 4 plots looking @ 5d fwds based on 5d traiing and seasonals, but each plot is filtered for NPD regime (low, low_mid, mid, high)
##fig4= same as fig3 but looks @21d fwds
##fig5= 6 plots looking @ diff fwds based on 5d trailing and seasonals in high NPD regime
##fig6= same as fig3 but uses trailing 21d instead of trailing 5d
##fig7= same as 6 but looks @ fwd 21 not fwd 5
##fig8= 4 plots looking @ both fwd 21 and fwd 5 based on both trailing 21 and trailing 5 all within high NPD regime.
##fig9= 6 plots same as 8 but adds two additional plots to look @1d fwds
##fig10= same as 9, but low mid NPD (less than -5 and greater than -10) filter rather than high.
##fig11= same as 8, but low NPD filter less than -10
##fig12= same as 8, but mid NPD filter (between -5 and 5)
##fig13= same structure as fig 8, but the z axis is VIX level not price movement in SPX
##fig14=same as 8 but low mid NPD filter (between -5 and -10)
##fig15 = same as 13 but uses mid NPD filter (between -5 and 5) rather than high NPD filter
##fig16 = same as 5 but mid NPD filter not high
##fig17 = 6 plots that show fwd 1,5,21d returns using 5 and 21d trailing as predictors (filtered for mid NPD regime)

# fig.show()
fig2.show()
# fig3.show()
# fig4.show()
# fig5.show()
# fig6.show()
# fig7.show()
# fig9.show()
# fig10.show()
# fig8.show()   ###high
# fig14.show()  ###low-mid
# fig11.show()  ###low
# fig12.show()  ###mid
# fig13.show()
# fig15.show()
# fig16.show()
# fig17.show()

if NPD_today < -6.75:
    st.plotly_chart(fig11)
elif NPD_today > -6.75:
    st.plotly_chart(fig8)
# print(NPD_today)

# pio.write_image(fig13,"op.pdf")
