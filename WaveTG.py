
###############################################################################
####################### WaveTG Tool ###########################################
###############################################################################

# Author: Alejandro Acedo Fajardo
###############################################################################

import math
import numpy as np
import matplotlib.pyplot as plt
import pywt as wt
import pandas as pd
from pywt import wavedec
from pywt import waverec
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import WeekdayLocator, MO, DateFormatter
import codecs

def generateHTMLfile (max_real_in, mean_real_in, max_real_out, mean_real_out, max_w_wavelet_in, mean_w_wavelet_in, max_w_wavelet_out, mean_w_wavelet_out, max_m_wavelet_in, mean_m_wavelet_in, max_m_wavelet_out, mean_m_wavelet_out, max_y_wavelet_in, mean_y_wavelet_in, max_y_wavelet_out, mean_y_wavelet_out):

	f = open('ResultsPage.html','w')

	part_1 = """<html>
	<head>
	<style>
	p.sansserif{
	   font-family: Arial, Helvetica, sans-serif;
	}
	</style>
	</head>

	<body style="background-color:#F3F3F3;">


	<br><br>

	<picture>
	<img src="DailyGraph.png">
	</picture>

	<br><br>
	<div class = 'my-legend'>
	<div class = 'legend-scale'>
	<ul class='legend-labels'>
	<li><span style='background:#00CC00';></span><p class="sansserif">Incoming Traffic in Bits per second &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Maximal In:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; """ 

	part_2 = str(max_real_in)
	part_3 = """ Gbps&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Average In: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""
	part_4 = str(mean_real_in)
	part_5 = """ Gbps</p></li>
	  <li><span style='background:#0000FF';></span><p class="sansserif">Outcoming Traffic in Bits per second&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Maximal Out:&nbsp;&nbsp; """
	part_6 = str(max_real_out)
	part_7 = """ Gbps&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Average Out:&nbsp;&nbsp;&nbsp;"""
	part_8 = str(mean_real_out)
	
	
	part_9 = """ Gbps</p></li>
	</ul>
	</div>
	</div>

	<style type='text/css'>
	.my-legend .legend-scale ul{
	margin:0;
	margin-bottom:5px;
	padding: 0;
	float: left;
	list-style: none;
	}
	.my-legend .legend-scale ul li{
	font-size:90%;
	list-style:none;
	margin-left: 150px;
	line-height: 18px;
	margin-bottom: 2px;
	width: 1000px;
	}
	.my-legend ul.legend-labels li span{
	display: block;
	float: left;
	height: 15px;
	width: 15px;
	margin-right: 5px;
	margin.left: 0;
	border: 1px solid #999;
	}
	</style>"""

	part_10 = """
	
	<br><br><br><br><br><br><br>
	<picture>
	<img src="WeeklyWavelet.png">
	</picture>
	<br><br>
	<div class = 'my-legend'>
	<div class = 'legend-scale'>
	<ul class='legend-labels'>
	<li><span style='background:#00CC00';></span><p class="sansserif">Incoming Traffic in Bits per second &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Maximal In:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; """
	part_11 = str(max_w_wavelet_in)
	part_12 = """ Gbps&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Average In: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""
	part_13 = str(mean_w_wavelet_in)
	part_14 = """ Gbps</p></li>
  <li><span style='background:#0000FF';></span><p class="sansserif">Outcoming Traffic in Bits per second&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Maximal Out:&nbsp;&nbsp; """
	part_15 = str(max_w_wavelet_out)
	part_16 = """ Gbps&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Average Out:&nbsp;&nbsp;&nbsp;&nbsp;"""
	part_17 = str(mean_w_wavelet_out)
	part_18 = """ Gbps</p></li>
	</ul>
	</div>
	</div>
	
	<br><br><br><br><br><br><br>
	<picture>
	<img src="MonthlyWavelet.png">
	</picture>
	<br><br>
	<div class = 'my-legend'>
	<div class = 'legend-scale'>
	<ul class='legend-labels'>
	<li><span style='background:#00CC00';></span><p class="sansserif">Incoming Traffic in Bits per second &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Maximal In:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; """
	part_19 = str(max_m_wavelet_in)
	part_20 = """ Gbps&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Average In: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""
	part_21 = str(mean_m_wavelet_in)
	part_22 = """ Gbps</p></li>
  <li><span style='background:#0000FF';></span><p class="sansserif">Outcoming Traffic in Bits per second&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Maximal Out:&nbsp;&nbsp; """
	part_23 = str(max_m_wavelet_out)
	part_24 = """ Gbps&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Average Out:&nbsp;&nbsp;&nbsp;"""
	part_25 = str(mean_m_wavelet_out)
	part_26 = """ Gbps</p></li>
	</ul>
	</div>
	</div>

	<br><br><br><br><br><br><br>
	<picture>
	<img src="YearlyWavelet.png">
	</picture>
	<br><br>
	<div class = 'my-legend'>
	<div class = 'legend-scale'>
	<ul class='legend-labels'>
	<li><span style='background:#00CC00';></span><p class="sansserif">Incoming Traffic in Bits per second &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Maximal In:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; """
	part_27 = str(max_y_wavelet_in)
	part_28 = """ Gbps&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Average In: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"""
	part_29 = str(mean_y_wavelet_in)
 	part_30 = """ Gbps</p></li>
  <li><span style='background:#0000FF';></span><p class="sansserif">Outcoming Traffic in Bits per second&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Maximal Out:&nbsp;&nbsp; """
	part_31 = str(max_y_wavelet_out)
	part_32 = """ Gbps&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Average Out:&nbsp;&nbsp;&nbsp;"""
	part_33 = str(mean_y_wavelet_out)
	part_34 = """ Gbps</p></li>
	</ul>
	</div>
	</div>"""
	
	final_part = """
	</body>
	</html>"""

	message = part_1 + part_2 + part_3 + part_4 + part_5 + part_6 + part_7 + part_8 + part_9 + part_10 + part_11 + part_12 + part_13 + part_14 + part_15 + part_16 + part_17 + part_18 + part_19 + part_20 + part_21 + part_22 + part_23 + part_24 + part_25 + part_26 + part_27 + part_28 + part_29 + part_30 + part_31 + part_32 + part_33 + part_34 + final_part

	f.write(message)
	f.close


def posTimestamp(matrix, timestamp):
    for i in range(0,len(matrix)):
    	if (matrix[i][0] == timestamp):       
	    return i
            break


def bufferWavelet (Matrix):


    array_timestamps = []
    signal_original_IN = []
    signal_original_OUT = []

    array_timestamps = Matrix[:,0].transpose()
    signal_original_IN = Matrix[:,1].transpose() 
    signal_original_OUT = Matrix[:,2].transpose()
 

    buffer_semanal_IN = [] #2016 samples
    buffer_semanal_OUT = [] #2016 samples

    buffer_mensual_IN = [] #8064 samples
    buffer_mensual_OUT = [] #8064 samples

    buffer_anual_IN = [] #96768 samples
    buffer_anual_OUT = [] #96768 samples


    bbdd_real_IN = [] # 672 positions
    bbdd_real_OUT = [] # 672 positions

    bbdd_semanal_wavelet_IN = [] #672 positions
    bbdd_semanal_wavelet_OUT = [] #672 positions
    bbdd_mensual_wavelet_IN = [] #672 positions
    bbdd_mensual_wavelet_OUT = [] #672 positions
    bbdd_anual_wavelet_IN = [] #670 positions
    bbdd_anual_wavelet_OUT = [] #670 positions

    bbdd_semanal_mrtg_IN = [] #672 positions
    bbdd_semanal_mrtg_OUT = [] #672 positions
    bbdd_mensual_mrtg_IN = [] #672 positions
    bbdd_mensual_mrtg_OUT = [] #672 positions
    bbdd_anual_mrtg_IN = [] #672 positions
    bbdd_anual_mrtg_OUT = [] #672 positions

    cont_real = 0
    cont_s = 0
    cont_m = 0
    cont_a = 0

    inicio_wavelet_semanal = 0
    inicio_wavelet_mensual = 0
    inicio_wavelet_anual = 0


    pos_value_s = 0
    pos_value_m = 0
    pos_value_a = 0



    for j in range(0,len(signal_original_IN)):
        
        cont_real = cont_real + 1
        cont_s = cont_s + 1
        cont_m = cont_m + 1
        cont_a = cont_a + 1

        # Real Traffic
        if (cont_real <= 672):
            bbdd_real_IN.append(signal_original_IN[j]) 
            bbdd_real_OUT.append(signal_original_OUT[j])

        else:
            bbdd_real_IN[0:670+1] = bbdd_real_IN[1:671+1]
            bbdd_real_OUT[0:670+1] = bbdd_real_OUT[1:671+1]
            bbdd_real_IN[671] = signal_original_IN[j]
            bbdd_real_OUT[671] = signal_original_OUT[j]
   

        # WAVELET
        if (cont_s == 2016):
            cont_s = 0

            buffer_semanal_IN = signal_original_IN[inicio_wavelet_semanal:j+1]
            buffer_semanal_OUT = signal_original_OUT[inicio_wavelet_semanal:j+1]

            inicio_wavelet_semanal = j + 1

            coeffs_IN = wavedec(buffer_semanal_IN, 'sym10', 'periodization',level=3)
            coeffs_OUT = wavedec(buffer_semanal_OUT, 'sym10', 'periodization',level=3)
            cA3_IN, cD3_IN, cD2_IN, cD1_IN = coeffs_IN
            cA3_OUT, cD3_OUT, cD2_OUT, cD1_OUT = coeffs_OUT

             
            # DETAILS
            vector_detalles_IN = []
            vector_detalles_OUT = []

            # Merge 3 arrays en only one
            vector_detalles_IN = np.hstack((cD3_IN,cD2_IN,cD1_IN)).ravel()
            vector_detalles_OUT = np.hstack((cD3_OUT,cD2_OUT,cD1_OUT)).ravel()

            vector_detalles_abs_IN = []
            vector_detalles_abs_OUT = []
            vector_detalles_abs_IN = np.absolute(vector_detalles_IN)
            vector_detalles_abs_OUT = np.absolute(vector_detalles_OUT)

            indices_IN = []
            indices_OUT = []
            indices_IN = sorted(range(len(vector_detalles_abs_IN)), key=lambda k: vector_detalles_abs_IN[k], reverse=True)
            indices_OUT = sorted(range(len(vector_detalles_abs_OUT)), key=lambda k: vector_detalles_abs_OUT[k], reverse=True)
            
            # indices corresponds to the positions of the array sorted by MAX to MEAN 
            # Difference of samples: 336(mrtg) - 252(wavelet) = 84, 42 details and 42 positions

            indices_IN = indices_IN[0:42]
            indices_OUT = indices_OUT[0:42]
            coefs_detalle_IN = []
            coefs_detalle_OUT = []

            for i in range(0,len(indices_IN)):             
                coefs_detalle_IN.append(vector_detalles_IN[indices_IN[i]])
                coefs_detalle_OUT.append(vector_detalles_OUT[indices_OUT[i]])


            # APROXIMATION + DETAILS
            
            coeficientes_a_d_IN = np.hstack((cA3_IN,coefs_detalle_IN,indices_IN)).ravel()
            coeficientes_a_d_OUT = np.hstack((cA3_OUT,coefs_detalle_OUT,indices_OUT)).ravel()
            n_coeficientes_semanal_IN = len(cA3_IN) + len(coefs_detalle_IN)
            n_coeficientes_semanal_OUT = len(cA3_OUT) + len(coefs_detalle_OUT)
            

            if (len(bbdd_semanal_wavelet_IN) != 672):
               
                bbdd_semanal_wavelet_IN = np.hstack((bbdd_semanal_wavelet_IN,coeficientes_a_d_IN)).ravel()
                bbdd_semanal_wavelet_OUT = np.hstack((bbdd_semanal_wavelet_OUT,coeficientes_a_d_OUT)).ravel()
             
            else:
                aux_IN = bbdd_semanal_wavelet_IN[336:671+1]
                aux_OUT = bbdd_semanal_wavelet_OUT[336:671+1]
                bbdd_semanal_wavelet_IN[0:335+1] = aux_IN
                bbdd_semanal_wavelet_OUT[0:335+1] = aux_OUT          
                bbdd_semanal_wavelet_IN[336:671+1] = coeficientes_a_d_IN
                bbdd_semanal_wavelet_OUT[336:671+1] = coeficientes_a_d_OUT


            # Reconstruction
            signal_reconstruida_semana_wavelet_IN = []
            signal_reconstruida_semana_wavelet_OUT = []
 
            for k in range(0, 1+1):
                cfx_new_IN = np.zeros(len(vector_detalles_IN))
                cfx_new_OUT = np.zeros(len(vector_detalles_OUT))
                
                if (k==0): # first week (k=0)
                    for t in range(1,len(coefs_detalle_IN)):
                        cfx_new_IN[int(bbdd_semanal_wavelet_IN[293+t])] = bbdd_semanal_wavelet_IN[251+t]
                        cfx_new_OUT[int(bbdd_semanal_wavelet_OUT[293+t])] = bbdd_semanal_wavelet_OUT[251+t]

		    cf_IN = np.hstack((bbdd_semanal_wavelet_IN[(k*336):(k*336+252)],cfx_new_IN)).ravel()
                    cf_OUT = np.hstack((bbdd_semanal_wavelet_OUT[(k*336):(k*336+252)],cfx_new_OUT)).ravel()
                    
		    cA3_r_IN = cf_IN[0:251+1]
                    cA3_r_OUT = cf_OUT[0:251+1]
                    cD3_r_IN = cf_IN[252:503+1]
                    cD3_r_OUT = cf_OUT[252:503+1]
                    cD2_r_IN = cf_IN[504:1007+1]
                    cD2_r_OUT = cf_OUT[504:1007+1]
                    cD1_r_IN = cf_IN[1008:2015+1]
                    cD1_r_OUT = cf_OUT[1008:2015+1]
                    
		    cf1_IN = []
                    cf1_OUT = []
                    cf1_IN.append(cA3_r_IN)
                    cf1_OUT.append(cA3_r_OUT)
                    cf1_IN.append(cD3_r_IN)
                    cf1_OUT.append(cD3_r_OUT)
		    cf1_IN.append(cD2_r_IN)
                    cf1_OUT.append(cD2_r_OUT)
		    cf1_IN.append(cD1_r_IN)
                    cf1_OUT.append(cD1_r_OUT)
                    

                    signal_reconstruida_semana_wavelet_IN = np.hstack((signal_reconstruida_semana_wavelet_IN, waverec(cf1_IN, 'sym10', 'periodization'))).ravel()

                    signal_reconstruida_semana_wavelet_OUT = np.hstack((signal_reconstruida_semana_wavelet_OUT, waverec(cf1_OUT, 'sym10', 'periodization'))).ravel()

                else: # second week (k=1)
                    if (len(bbdd_semanal_wavelet_IN)>336):

                        for t in range(1,len(coefs_detalle_IN)):
                            cfx_new_IN[int(bbdd_semanal_wavelet_IN[629+t])] = bbdd_semanal_wavelet_IN[587+t]
                            cfx_new_OUT[int(bbdd_semanal_wavelet_OUT[629+t])] = bbdd_semanal_wavelet_OUT[587+t]

                        cf_IN = np.hstack((bbdd_semanal_wavelet_IN[(k*336):(k*336+252)],cfx_new_IN)).ravel()
                        cf_OUT = np.hstack((bbdd_semanal_wavelet_OUT[(k*336):(k*336+252)],cfx_new_OUT)).ravel()
                        cA3_r_IN = cf_IN[0:251+1]
                        cA3_r_OUT = cf_OUT[0:251+1]
	            	cD3_r_IN = cf_IN[252:503+1]
                        cD3_r_OUT = cf_OUT[252:503+1]
	            	cD2_r_IN = cf_IN[504:1007+1]
                        cD2_r_OUT = cf_OUT[504:1007+1]
	           	cD1_r_IN = cf_IN[1008:2015+1]
                        cD1_r_OUT = cf_OUT[1008:2015+1]
	            
		    	cf1_IN = []
                        cf1_OUT = []
	            	cf1_IN.append(cA3_r_IN)
                        cf1_OUT.append(cA3_r_OUT)
	            	cf1_IN.append(cD3_r_IN)
                        cf1_OUT.append(cD3_r_OUT)
		    	cf1_IN.append(cD2_r_IN)
                        cf1_OUT.append(cD2_r_OUT)
		    	cf1_IN.append(cD1_r_IN)
                        cf1_OUT.append(cD1_r_OUT)

                        signal_reconstruida_semana_wavelet_IN = np.hstack((signal_reconstruida_semana_wavelet_IN, waverec(cf1_IN, 'sym10', 'periodization'))).ravel()
                        signal_reconstruida_semana_wavelet_OUT = np.hstack((signal_reconstruida_semana_wavelet_OUT, waverec(cf1_OUT, 'sym10', 'periodization'))).ravel()

                        for p in range(0,len(signal_reconstruida_semana_wavelet_IN)):
                           if (signal_reconstruida_semana_wavelet_IN[p] < 0):
                               signal_reconstruida_semana_wavelet_IN[p] = 0
                           if (signal_reconstruida_semana_wavelet_OUT[p] < 0):
                               signal_reconstruida_semana_wavelet_OUT[p] = 0
 
                        
            # In orde to compare
            # We take the corresponding interval of the signal
            if (j == 2015):
                signal_original_semana_IN = signal_original_IN[0:2015+1]
                signal_original_semana_OUT = signal_original_OUT[0:2015+1]
            elif (j == 4031):
                signal_original_semana_IN = signal_original_IN[0:4031+1]
                signal_original_semana_OUT = signal_original_OUT[0:4031+1]
            else:
                pos_value_s = pos_value_s + 1
                pos_inf = pos_value_s*2016
                pos_sup = (pos_value_s + 2)*2016
                signal_original_semana_IN = signal_original_IN[pos_inf:pos_sup]
                signal_original_semana_OUT = signal_original_OUT[pos_inf:pos_sup]

            if(j == 193535):

                # Real Traffic
                bbdd_real_IN = [8 * float(x) / 1000000000 for x in bbdd_real_IN]
                bbdd_real_OUT = [8 * float(x) / 1000000000 for x in bbdd_real_OUT]

		max_real_in = round(np.amax(bbdd_real_IN),3)
		max_real_out = round(np.amax(bbdd_real_OUT),3)
		mean_real_in = round(np.mean(bbdd_real_IN),3)
		mean_real_out = round(np.mean(bbdd_real_OUT),3)

                w=2
	        h=len(bbdd_real_IN)
                Matrix_real = np.array([[0 for x in range(w)] for y in range(h)], dtype='f')

                Matrix_real[:,0] = bbdd_real_IN
                Matrix_real[:,1] = bbdd_real_OUT
                
                import datetime as dt
                import pytz
            
                tz = pytz.timezone('Europe/Madrid')
                fecha = dt.datetime.utcfromtimestamp(1235142000).replace(tzinfo=pytz.utc)
                dt = tz.normalize(fecha.astimezone(tz))
                fecha = dt.strftime('%Y-%m-%d %H:%M')
 
	        array_dates_day = pd.date_range(fecha, periods = len(bbdd_real_IN), freq = '300S')
	        ts_day = pd.Series(bbdd_real_IN, index=array_dates_day)

                fig = plt.figure()
                fig.suptitle('Input / Output Real Traffic', fontsize=14, fontweight='bold')
	        df = pd.DataFrame(Matrix_real, index=ts_day.index, columns=['Input Mbs','Output Mbs'])
                ax = df.plot(figsize=(10.5,3.5),color=['#00cc00', '#0000ff'], title='Daily Graph (5 Minute Average)',lw=1,colormap='jet');
                ax.fill_between(ts_day.index,0,Matrix_real[:,0], facecolor='#00cc00')  
                ax.set_ylabel("Bits per Second")
                ax.legend().set_visible(False)
                labels_y_daily = ax.yaxis.get_majorticklocs()

                longitud = len(labels_y_daily)

                new_labels_y_daily = []
                for e in range(0,len(labels_y_daily)):
                    labels_y_daily[e]=format(labels_y_daily[e], '.1f')
                    new_labels_y_daily.append('%.1f G' %labels_y_daily[e])   
                ax.yaxis.set_ticklabels(np.array(new_labels_y_daily))
            
                ax.grid(b=True, which='both',color='k', linestyle=':',alpha=0.7)
                fig.patch.set_facecolor('#F3F3F3')
                plt.margins(x=0)
            
                plt.savefig('DailyGraph.png', facecolor=fig.get_facecolor())
                plt.savefig('DailyGraph.eps', facecolor=fig.get_facecolor())
                # Real Traffic


                signal_reconstruida_semana_wavelet_Mbs_IN = [8 * float(x) / 1000000000 for x in signal_reconstruida_semana_wavelet_IN]
                signal_reconstruida_semana_wavelet_Mbs_OUT = [8 * float(x) / 1000000000 for x in signal_reconstruida_semana_wavelet_OUT]

	        signal_reconstruida_semana_wavelet_Mbs_IN = np.array(signal_reconstruida_semana_wavelet_Mbs_IN)
                signal_reconstruida_semana_wavelet_Mbs_OUT = np.array(signal_reconstruida_semana_wavelet_Mbs_OUT)


		max_w_wavelet_in = round(np.amax(signal_reconstruida_semana_wavelet_Mbs_IN),3)
		max_w_wavelet_out = round(np.amax(signal_reconstruida_semana_wavelet_Mbs_OUT),3)
		mean_w_wavelet_in = round(np.mean(signal_reconstruida_semana_wavelet_Mbs_IN),3)
		mean_w_wavelet_out = round(np.mean(signal_reconstruida_semana_wavelet_Mbs_OUT),3)

		import datetime as dt
                import pytz

		tz = pytz.timezone('Europe/Madrid')
                fecha = dt.datetime.utcfromtimestamp(1234134000).replace(tzinfo=pytz.utc)
                dt = tz.normalize(fecha.astimezone(tz))
                fecha = dt.strftime('%Y-%m-%d %H:%M')

	        array_dates_week = pd.date_range(fecha, periods = len(signal_reconstruida_semana_wavelet_Mbs_IN), freq = '300S')
	        ts_week = pd.Series(signal_reconstruida_semana_wavelet_Mbs_IN, index=array_dates_week)

                w=2
	        h=len(signal_reconstruida_semana_wavelet_Mbs_IN)
                Matrix_r_wavelet = np.array([[0 for x in range(w)] for y in range(h)], dtype='f')

                Matrix_r_wavelet[:,0] = signal_reconstruida_semana_wavelet_Mbs_IN.transpose()
                Matrix_r_wavelet[:,1] = signal_reconstruida_semana_wavelet_Mbs_OUT.transpose()
          
                # Wavelet
                fig = plt.figure()
                fig.suptitle('Input / Output Traffic Wavelet', fontsize=14, fontweight='bold')
	        df = pd.DataFrame(Matrix_r_wavelet, index=ts_week.index, columns=['Input Mbs','Output Mbs'])
                ax = df.plot(figsize=(10.5,3.5),color=['#00cc00', '#0000ff'], title='Weekly Wavelet Graph (Level 3, Sym10)',lw=1,colormap='jet');
                ax.fill_between(ts_week.index,0,Matrix_r_wavelet[:,0], facecolor='#00cc00')  

                ax.set_ylabel("Bits per Second")
                ax.legend().set_visible(False)
                labels_y_weekly = ax.yaxis.get_majorticklocs()
                new_labels_y_weekly = []
                for e in range(0,len(labels_y_weekly)):
                    labels_y_weekly[e]=format(labels_y_weekly[e], '.1f')
                    new_labels_y_weekly.append('%.1f G' %labels_y_weekly[e])   
                ax.yaxis.set_ticklabels(np.array(new_labels_y_weekly))
                ax.grid(b=True, which='both',color='k', linestyle=':',alpha=0.7)

                fig.patch.set_facecolor('#F3F3F3')
                plt.margins(x=0)
                plt.savefig('WeeklyWavelet.png',facecolor=fig.get_facecolor())
                plt.savefig('WeeklyWavelet.eps',facecolor=fig.get_facecolor())

                fig.clear()
                    
        if (cont_m == 8064):
            cont_m = 0
            buffer_mensual_IN = signal_original_IN[inicio_wavelet_mensual:j+1]
            buffer_mensual_OUT = signal_original_OUT[inicio_wavelet_mensual:j+1]
            inicio_wavelet_mensual = j + 1

            coeffs_IN = wavedec(buffer_mensual_IN, 'sym12', 'periodization',level=5)
            coeffs_OUT = wavedec(buffer_mensual_OUT, 'sym12', 'periodization',level=5)
            cA5_IN, cD5_IN, cD4_IN, cD3_IN, cD2_IN, cD1_IN = coeffs_IN  
            cA5_OUT, cD5_OUT, cD4_OUT, cD3_OUT, cD2_OUT, cD1_OUT = coeffs_OUT            
   
            # DETAILS
            vector_detalles_IN = []
            vector_detalles_OUT = []
            # Merge of 5 arrays in only 1 
            vector_detalles_IN = np.hstack((cD5_IN,cD4_IN,cD3_IN,cD2_IN,cD1_IN)).ravel()
            vector_detalles_OUT = np.hstack((cD5_OUT,cD4_OUT,cD3_OUT,cD2_OUT,cD1_OUT)).ravel()

            vector_detalles_abs_IN = []
            vector_detalles_abs_OUT = []
            vector_detalles_abs_IN = np.absolute(vector_detalles_IN)
            vector_detalles_abs_OUT = np.absolute(vector_detalles_OUT)

            indices_IN = []
            indices_OUT = []
            indices_IN = sorted(range(len(vector_detalles_abs_IN)), key=lambda k: vector_detalles_abs_IN[k], reverse=True)
            indices_OUT = sorted(range(len(vector_detalles_abs_OUT)), key=lambda k: vector_detalles_abs_OUT[k], reverse=True)
            
            # indices corresponds to the positions of the array sorted by MAX to MEAN 
            # Difference of samples: 336(mrtg) - 252(wavelet) = 84, 42 details y 42 positions

            indices_IN = indices_IN[0:42]
            indices_OUT = indices_OUT[0:42]
            coefs_detalle_IN = []
            coefs_detalle_OUT = []

            for i in range(0,len(indices_IN)):             
                coefs_detalle_IN.append(vector_detalles_IN[indices_IN[i]])
                coefs_detalle_OUT.append(vector_detalles_OUT[indices_OUT[i]])


            # APROXIMATION + DETAILS

            coeficientes_a_d_IN = np.hstack((cA5_IN,coefs_detalle_IN,indices_IN)).ravel()
            coeficientes_a_d_OUT = np.hstack((cA5_OUT,coefs_detalle_OUT,indices_OUT)).ravel()
            n_coeficientes_mensual_IN = len(cA5_IN) + len(coefs_detalle_IN)
            n_coeficientes_mensual_OUT = len(cA5_OUT) + len(coefs_detalle_OUT)
            

            if (len(bbdd_mensual_wavelet_IN) != 672):
               
                bbdd_mensual_wavelet_IN = np.hstack((bbdd_mensual_wavelet_IN,coeficientes_a_d_IN)).ravel()
                bbdd_mensual_wavelet_OUT = np.hstack((bbdd_mensual_wavelet_OUT,coeficientes_a_d_OUT)).ravel()
             
            else:
                aux_IN = bbdd_mensual_wavelet_IN[336:671+1]
                aux_OUT = bbdd_mensual_wavelet_OUT[336:671+1]
                bbdd_mensual_wavelet_IN[0:335+1] = aux_IN
                bbdd_mensual_wavelet_OUT[0:335+1] = aux_OUT        
                bbdd_mensual_wavelet_IN[336:671+1] = coeficientes_a_d_IN
                bbdd_mensual_wavelet_OUT[336:671+1] = coeficientes_a_d_OUT

            # Reconstruction
            signal_reconstruida_mes_wavelet_IN = []
            signal_reconstruida_mes_wavelet_OUT = []
 
            for k in range(0, 1+1):
                cfx_new_IN = np.zeros(len(vector_detalles_IN))
                cfx_new_OUT = np.zeros(len(vector_detalles_OUT))
                
                if (k==0): # first month (k=0)
                    for t in range(1,len(coefs_detalle_IN)):
                        cfx_new_IN[int(bbdd_mensual_wavelet_IN[293+t])] = bbdd_mensual_wavelet_IN[251+t]
                        cfx_new_OUT[int(bbdd_mensual_wavelet_OUT[293+t])] = bbdd_mensual_wavelet_OUT[251+t]

		    cf_IN = np.hstack((bbdd_mensual_wavelet_IN[(k*336):(k*336+252)],cfx_new_IN)).ravel()
                    cf_OUT = np.hstack((bbdd_mensual_wavelet_OUT[(k*336):(k*336+252)],cfx_new_OUT)).ravel()
                    

		    cA5_r_IN = cf_IN[0:251+1]
                    cA5_r_OUT = cf_OUT[0:251+1]
                    cD5_r_IN = cf_IN[252:503+1]
                    cD5_r_OUT = cf_OUT[252:503+1]
                    cD4_r_IN = cf_IN[504:1007+1]
                    cD4_r_OUT = cf_OUT[504:1007+1]
                    cD3_r_IN = cf_IN[1008:2015+1]
                    cD3_r_OUT = cf_OUT[1008:2015+1]
                    cD2_r_IN = cf_IN[2016:4031+1]
                    cD2_r_OUT = cf_OUT[2016:4031+1]
                    cD1_r_IN = cf_IN[4032:8063+1]
                    cD1_r_OUT = cf_OUT[4032:8063+1]
                    
		    cf1_IN = []
                    cf1_OUT = []
                    cf1_IN.append(cA5_r_IN)
 	            cf1_OUT.append(cA5_r_OUT)
                    cf1_IN.append(cD5_r_IN)
                    cf1_OUT.append(cD5_r_OUT)
		    cf1_IN.append(cD4_r_IN)
                    cf1_OUT.append(cD4_r_OUT)
		    cf1_IN.append(cD3_r_IN)
                    cf1_OUT.append(cD3_r_OUT)
		    cf1_IN.append(cD2_r_IN)
                    cf1_OUT.append(cD2_r_OUT)
                    cf1_IN.append(cD1_r_IN)
                    cf1_OUT.append(cD1_r_OUT)
                    
                    signal_reconstruida_mes_wavelet_IN = np.hstack((signal_reconstruida_mes_wavelet_IN, waverec(cf1_IN, 'sym12', 'periodization'))).ravel()

                    signal_reconstruida_mes_wavelet_OUT = np.hstack((signal_reconstruida_mes_wavelet_OUT, waverec(cf1_OUT, 'sym12', 'periodization'))).ravel()

                    for p in range(0,len(signal_reconstruida_mes_wavelet_IN)):
                        if (signal_reconstruida_mes_wavelet_IN[p] < 0):
                            signal_reconstruida_mes_wavelet_IN[p] = 0
                        if (signal_reconstruida_mes_wavelet_OUT[p] < 0):
                            signal_reconstruida_mes_wavelet_OUT[p] = 0

                else: # second month (k=1)
                    if (len(bbdd_mensual_wavelet_IN)>336):

                        for t in range(1,len(coefs_detalle_IN)):
                            cfx_new_IN[int(bbdd_mensual_wavelet_IN[629+t])] = bbdd_mensual_wavelet_IN[587+t]
                            cfx_new_OUT[int(bbdd_mensual_wavelet_OUT[629+t])] = bbdd_mensual_wavelet_OUT[587+t]

                        cf_IN = np.hstack((bbdd_mensual_wavelet_IN[(k*336):(k*336+252)],cfx_new_IN)).ravel()
                        cf_OUT = np.hstack((bbdd_mensual_wavelet_OUT[(k*336):(k*336+252)],cfx_new_OUT)).ravel()
                        
		    	cA5_r_IN = cf_IN[0:251+1]
                        cA5_r_OUT = cf_OUT[0:251+1]
	            	cD5_r_IN = cf_IN[252:503+1]
                        cD5_r_OUT = cf_OUT[252:503+1]
	            	cD4_r_IN = cf_IN[504:1007+1]
                        cD4_r_OUT = cf_OUT[504:1007+1]
	            	cD3_r_IN = cf_IN[1008:2015+1]
                        cD3_r_OUT = cf_OUT[1008:2015+1]
	            	cD2_r_IN = cf_IN[2016:4031+1]
                        cD2_r_OUT = cf_OUT[2016:4031+1]
	            	cD1_r_IN = cf_IN[4032:8063+1]
                        cD1_r_OUT = cf_OUT[4032:8063+1]
	            
		    	cf1_IN = []
                        cf1_OUT = []
	            	cf1_IN.append(cA5_r_IN)
                        cf1_OUT.append(cA5_r_OUT)
	            	cf1_IN.append(cD5_r_IN)
                        cf1_OUT.append(cD5_r_OUT)
		    	cf1_IN.append(cD4_r_IN)
                        cf1_OUT.append(cD4_r_OUT)
		    	cf1_IN.append(cD3_r_IN)
                        cf1_OUT.append(cD3_r_OUT)
		    	cf1_IN.append(cD2_r_IN)
                        cf1_OUT.append(cD2_r_OUT)
	            	cf1_IN.append(cD1_r_IN)
                        cf1_OUT.append(cD1_r_OUT)

                        signal_reconstruida_mes_wavelet_IN = np.hstack((signal_reconstruida_mes_wavelet_IN, waverec(cf1_IN, 'sym12', 'periodization'))).ravel()

                        signal_reconstruida_mes_wavelet_OUT = np.hstack((signal_reconstruida_mes_wavelet_OUT, waverec(cf1_OUT, 'sym12', 'periodization'))).ravel()

         
            #We take the corresponding interval of the signal
            if (j == 8063):
                signal_original_mes_IN = signal_original_IN[0:8063+1]
                signal_original_mes_OUT = signal_original_OUT[0:8063+1]
            elif (j == 16127):
                signal_original_mes_IN = signal_original_IN[0:16127+1]
                signal_original_mes_OUT = signal_original_OUT[0:16127+1]
            else:
                pos_value_m = pos_value_m + 1
                pos_inf = pos_value_m*8064
                pos_sup = (pos_value_m + 2)*8064
                signal_original_mes_IN = signal_original_IN[pos_inf:pos_sup]
                signal_original_mes_OUT = signal_original_OUT[pos_inf:pos_sup]

            if(j == 193535):

                signal_reconstruida_mes_wavelet_Mbs_IN = [8 * float(x) / 1000000000 for x in signal_reconstruida_mes_wavelet_IN]
                signal_reconstruida_mes_wavelet_Mbs_OUT = [8 * float(x) / 1000000000 for x in signal_reconstruida_mes_wavelet_OUT]

                signal_reconstruida_mes_wavelet_Mbs_IN = np.array(signal_reconstruida_mes_wavelet_Mbs_IN)
                signal_reconstruida_mes_wavelet_Mbs_OUT = np.array(signal_reconstruida_mes_wavelet_Mbs_OUT)


		max_m_wavelet_in = round(np.amax(signal_reconstruida_mes_wavelet_Mbs_IN),3)
		max_m_wavelet_out = round(np.amax(signal_reconstruida_mes_wavelet_Mbs_OUT),3)
		mean_m_wavelet_in = round(np.mean(signal_reconstruida_mes_wavelet_Mbs_IN),3)
		mean_m_wavelet_out = round(np.mean(signal_reconstruida_mes_wavelet_Mbs_OUT),3)

                import datetime as dt
                import pytz

		tz = pytz.timezone('Europe/Madrid')
                fecha = dt.datetime.utcfromtimestamp(1230504900).replace(tzinfo=pytz.utc)
                dt = tz.normalize(fecha.astimezone(tz))
                fecha = dt.strftime('%Y-%m-%d %H:%M')

                array_dates_month = pd.date_range(fecha, periods = len(signal_reconstruida_mes_wavelet_Mbs_IN), freq = '300S')

	        ts_month = pd.Series(signal_reconstruida_mes_wavelet_Mbs_IN, index=array_dates_month)

                w=2
	        h=len(signal_reconstruida_mes_wavelet_Mbs_IN)
                Matrix_r_wavelet_month = np.array([[0 for x in range(w)] for y in range(h)], dtype='f')

	        Matrix_r_wavelet_month[:,0] = signal_reconstruida_mes_wavelet_Mbs_IN.transpose()
                Matrix_r_wavelet_month[:,1] = signal_reconstruida_mes_wavelet_Mbs_OUT.transpose()

                # Wavelet
                fig = plt.figure()
                fig.suptitle('Input / Output Traffic Wavelet', fontsize=14, fontweight='bold')
	        df = pd.DataFrame(Matrix_r_wavelet_month, index=ts_month.index, columns=['Input Mbs','Output Mbs'])

                ax = df.plot(figsize=(10.5,3.5),color=['#00cc00', '#0000ff'], title='Monthly Wavelet Graph (Level 5, Sym12)',lw=1,colormap='jet');
                ax.fill_between(ts_month.index,0,Matrix_r_wavelet_month[:,0], facecolor='#00cc00')  

                ax.set_ylabel("Bits per Second")
                ax.legend().set_visible(False)
                labels_y_monthly = ax.yaxis.get_majorticklocs()
                new_labels_y_monthly = []
                for e in range(0,len(labels_y_monthly)):
                    labels_y_monthly[e]=format(labels_y_monthly[e], '.1f')
                    new_labels_y_monthly.append('%.1f G' %labels_y_monthly[e])   
                ax.yaxis.set_ticklabels(np.array(new_labels_y_monthly))

                ax.grid(b=True, which='both',color='k', linestyle=':',alpha=0.7)

                fig.patch.set_facecolor('#F3F3F3')
                plt.margins(x=0)
                plt.savefig('MonthlyWavelet.png',facecolor=fig.get_facecolor())
                plt.savefig('MonthlyWavelet.eps',facecolor=fig.get_facecolor())

                fig.clear()

        if (cont_a == 96768):
  
            cont_a = 0
            buffer_anual_IN = signal_original_IN[inicio_wavelet_anual:j+1]
            buffer_anual_OUT = signal_original_OUT[inicio_wavelet_anual:j+1]
            inicio_wavelet_anual = j + 1

            coeffs_IN = wavedec(buffer_anual_IN, 'sym17', 'periodization',level=9)
            coeffs_OUT = wavedec(buffer_anual_OUT, 'sym17', 'periodization',level=9)
            cA9_IN, cD9_IN, cD8_IN, cD7_IN, cD6_IN, cD5_IN, cD4_IN, cD3_IN, cD2_IN, cD1_IN = coeffs_IN        
            cA9_OUT, cD9_OUT, cD8_OUT, cD7_OUT, cD6_OUT, cD5_OUT, cD4_OUT, cD3_OUT, cD2_OUT, cD1_OUT = coeffs_OUT      
   
            # DETAILS
            vector_detalles_IN = []
	    vector_detalles_OUT = []
            # Merge of 9 arrays en only 1
            vector_detalles_IN = np.hstack((cD9_IN,cD8_IN,cD7_IN,cD6_IN,cD5_IN,cD4_IN,cD3_IN,cD2_IN,cD1_IN)).ravel()
            vector_detalles_OUT = np.hstack((cD9_OUT,cD8_OUT,cD7_OUT,cD6_OUT,cD5_OUT,cD4_OUT,cD3_OUT,cD2_OUT,cD1_OUT)).ravel()

            vector_detalles_abs_IN = []
            vector_detalles_abs_OUT = []
            vector_detalles_abs_IN = np.absolute(vector_detalles_IN)
            vector_detalles_abs_OUT = np.absolute(vector_detalles_OUT)

            indices_IN = []
            indices_OUT = []
            indices_IN = sorted(range(len(vector_detalles_abs_IN)), key=lambda k: vector_detalles_abs_IN[k], reverse=True)
            indices_OUT = sorted(range(len(vector_detalles_abs_OUT)), key=lambda k: vector_detalles_abs_OUT[k], reverse=True)
            
            # indices corresponds to the positions of the array sorted by MAX to MEAN 
            # Difference of samples: 336(mrtg) - 189(wavelet) = 147, 73 details y 73 positions

            indices_IN = indices_IN[0:73]
            indices_OUT = indices_OUT[0:73]
            coefs_detalle_IN = []
            coefs_detalle_OUT = []

            for i in range(0,len(indices_IN)):             
                coefs_detalle_IN.append(vector_detalles_IN[indices_IN[i]])
                coefs_detalle_OUT.append(vector_detalles_OUT[indices_OUT[i]])


            # APROXIMATION + DETAILS

            coeficientes_a_d_IN = np.hstack((cA9_IN,coefs_detalle_IN,indices_IN)).ravel()
            coeficientes_a_d_OUT = np.hstack((cA9_OUT,coefs_detalle_OUT,indices_OUT)).ravel()
            n_coeficientes_anual_IN = len(cA9_IN) + len(coefs_detalle_IN)
            n_coeficientes_anual_OUT = len(cA9_OUT) + len(coefs_detalle_OUT)
            

            if (len(bbdd_anual_wavelet_IN) != 670):
               
                bbdd_anual_wavelet_IN = np.hstack((bbdd_anual_wavelet_IN,coeficientes_a_d_IN)).ravel()
                bbdd_anual_wavelet_OUT = np.hstack((bbdd_anual_wavelet_OUT,coeficientes_a_d_OUT)).ravel()
             
            else:
                aux_IN = bbdd_anual_wavelet_IN[335:669+1]
                aux_OUT = bbdd_anual_wavelet_OUT[335:669+1]
                bbdd_anual_wavelet_IN[0:334+1] = aux_IN
                bbdd_anual_wavelet_OUT[0:334+1] = aux_OUT        
                bbdd_anual_wavelet_IN[335:669+1] = coeficientes_a_d_IN
                bbdd_anual_wavelet_OUT[335:669+1] = coeficientes_a_d_OUT

            # Reconstruction
            signal_reconstruida_anio_wavelet_IN = []
            signal_reconstruida_anio_wavelet_OUT = []
 
            for k in range(0, 1+1):
                cfx_new_IN = np.zeros(len(vector_detalles_IN))
                cfx_new_OUT = np.zeros(len(vector_detalles_OUT))
                
                if (k==0): # first year (k=0)
                    for t in range(1,len(coefs_detalle_IN)):
                        cfx_new_IN[int(bbdd_anual_wavelet_IN[261+t])] = bbdd_anual_wavelet_IN[188+t]
                        cfx_new_OUT[int(bbdd_anual_wavelet_OUT[261+t])] = bbdd_anual_wavelet_OUT[188+t]

		    cf_IN = np.hstack((bbdd_anual_wavelet_IN[(k*335):(k*335+189)],cfx_new_IN)).ravel()
                    cf_OUT = np.hstack((bbdd_anual_wavelet_OUT[(k*335):(k*335+189)],cfx_new_OUT)).ravel()
                   
		    cA9_r_IN = cf_IN[0:188+1]
                    cA9_r_OUT = cf_OUT[0:188+1]
                    cD9_r_IN = cf_IN[189:377+1]
                    cD9_r_OUT = cf_OUT[189:377+1]
                    cD8_r_IN = cf_IN[378:755+1]
                    cD8_r_OUT = cf_OUT[378:755+1]
                    cD7_r_IN = cf_IN[756:1511+1]
		    cD7_r_OUT = cf_OUT[756:1511+1]
                    cD6_r_IN = cf_IN[1512:3023+1] 
		    cD6_r_OUT = cf_OUT[1512:3023+1]
                    cD5_r_IN = cf_IN[3024:6047+1]
		    cD5_r_OUT = cf_OUT[3024:6047+1]
		    cD4_r_IN = cf_IN[6048:12095+1]
		    cD4_r_OUT = cf_OUT[6048:12095+1]
		    cD3_r_IN = cf_IN[12096:24191+1]
                    cD3_r_OUT = cf_OUT[12096:24191+1]
		    cD2_r_IN = cf_IN[24192:48383+1]
		    cD2_r_OUT = cf_OUT[24192:48383+1]
		    cD1_r_IN = cf_IN[48384:96767+1]
                    cD1_r_OUT = cf_OUT[48384:96767+1]

		    cf1_IN = []
                    cf1_OUT = []
                    cf1_IN.append(cA9_r_IN)
                    cf1_OUT.append(cA9_r_OUT)
                    cf1_IN.append(cD9_r_IN)
                    cf1_OUT.append(cD9_r_OUT)
		    cf1_IN.append(cD8_r_IN)
                    cf1_OUT.append(cD8_r_OUT)
		    cf1_IN.append(cD7_r_IN)
                    cf1_OUT.append(cD7_r_OUT)
		    cf1_IN.append(cD6_r_IN)
                    cf1_OUT.append(cD6_r_OUT)
                    cf1_IN.append(cD5_r_IN)
                    cf1_OUT.append(cD5_r_OUT)
		    cf1_IN.append(cD4_r_IN)
                    cf1_OUT.append(cD4_r_OUT)
		    cf1_IN.append(cD3_r_IN)
                    cf1_OUT.append(cD3_r_OUT)
		    cf1_IN.append(cD2_r_IN)
                    cf1_OUT.append(cD2_r_OUT)
		    cf1_IN.append(cD1_r_IN)
                    cf1_OUT.append(cD1_r_OUT)
                  
                    signal_reconstruida_anio_wavelet_IN = np.hstack((signal_reconstruida_anio_wavelet_IN, waverec(cf1_IN, 'sym17', 'periodization'))).ravel()

                    signal_reconstruida_anio_wavelet_OUT = np.hstack((signal_reconstruida_anio_wavelet_OUT, waverec(cf1_OUT, 'sym17', 'periodization'))).ravel()

                    for p in range(0,len(signal_reconstruida_anio_wavelet_IN)):
                        if (signal_reconstruida_anio_wavelet_IN[p] < 0):
                            signal_reconstruida_anio_wavelet_IN[p] = 0
                        if (signal_reconstruida_anio_wavelet_OUT[p] < 0):
                            signal_reconstruida_anio_wavelet_OUT[p] = 0


                else: # second year (k=1)
                    if (len(bbdd_anual_wavelet_IN)>335):

                        for t in range(1,len(coefs_detalle_IN)):
                            cfx_new_IN[int(bbdd_anual_wavelet_IN[596+t])] = bbdd_anual_wavelet_IN[523+t]
                            cfx_new_OUT[int(bbdd_anual_wavelet_OUT[596+t])] = bbdd_anual_wavelet_OUT[523+t]

                        cf_IN = np.hstack((bbdd_anual_wavelet_IN[(k*335):(k*335+189)],cfx_new_IN)).ravel()
                        cf_OUT = np.hstack((bbdd_anual_wavelet_OUT[(k*335):(k*335+189)],cfx_new_OUT)).ravel()
                        
		    cA9_r_IN = cf_IN[0:188+1]
                    cA9_r_OUT = cf_OUT[0:188+1]
                    cD9_r_IN = cf_IN[189:377+1]
                    cD9_r_OUT = cf_OUT[189:377+1]
                    cD8_r_IN = cf_IN[378:755+1]
                    cD8_r_OUT = cf_OUT[378:755+1]
                    cD7_r_IN = cf_IN[756:1511+1]
                    cD7_r_OUT = cf_OUT[756:1511+1]
                    cD6_r_IN = cf_IN[1512:3023+1]
                    cD6_r_OUT = cf_OUT[1512:3023+1]
                    cD5_r_IN = cf_IN[3024:6047+1]
                    cD5_r_OUT = cf_OUT[3024:6047+1]
		    cD4_r_IN = cf_IN[6048:12095+1]
                    cD4_r_OUT = cf_OUT[6048:12095+1]
		    cD3_r_IN = cf_IN[12096:24191+1]
                    cD3_r_OUT = cf_OUT[12096:24191+1]
		    cD2_r_IN = cf_IN[24192:48383+1]
                    cD2_r_OUT = cf_OUT[24192:48383+1]
		    cD1_r_IN = cf_IN[48384:96767+1]
                    cD1_r_OUT = cf_OUT[48384:96767+1]
                 
		    cf1_IN = []
                    cf1_OUT = []
                    cf1_IN.append(cA9_r_IN)
		    cf1_OUT.append(cA9_r_OUT)
                    cf1_IN.append(cD9_r_IN)
                    cf1_OUT.append(cD9_r_OUT)
		    cf1_IN.append(cD8_r_IN)
                    cf1_OUT.append(cD8_r_OUT)
		    cf1_IN.append(cD7_r_IN)
                    cf1_OUT.append(cD7_r_OUT)
		    cf1_IN.append(cD6_r_IN)
                    cf1_OUT.append(cD6_r_OUT)
                    cf1_IN.append(cD5_r_IN)
		    cf1_OUT.append(cD5_r_OUT)
		    cf1_IN.append(cD4_r_IN)
                    cf1_OUT.append(cD4_r_OUT)
		    cf1_IN.append(cD3_r_IN)
                    cf1_OUT.append(cD3_r_OUT)
		    cf1_IN.append(cD2_r_IN)
                    cf1_OUT.append(cD2_r_OUT)
		    cf1_IN.append(cD1_r_IN)
                    cf1_OUT.append(cD1_r_OUT)

                    signal_reconstruida_anio_wavelet_IN = np.hstack((signal_reconstruida_anio_wavelet_IN, waverec(cf1_IN, 'sym17', 'periodization'))).ravel()
                    signal_reconstruida_anio_wavelet_OUT = np.hstack((signal_reconstruida_anio_wavelet_OUT, waverec(cf1_OUT, 'sym17', 'periodization'))).ravel()

                    for p in range(0,len(signal_reconstruida_anio_wavelet_IN)):
                        if (signal_reconstruida_anio_wavelet_IN[p] < 0):
                            signal_reconstruida_anio_wavelet_IN[p] = 0
                        if (signal_reconstruida_anio_wavelet_OUT[p] < 0):
                            signal_reconstruida_anio_wavelet_OUT[p] = 0
 
                        
            #We take the corresponding interval of the signal
            if (j == 96767):
                signal_original_anio_IN = signal_original_IN[0:96767+1]
                signal_original_anio_OUT = signal_original_OUT[0:96767+1]
            elif (j == 193535):
                signal_original_anio_IN = signal_original_IN[0:193535+1]
                signal_original_anio_OUT = signal_original_OUT[0:193535+1]
            else:
                pos_value_a = pos_value_a + 1
                pos_inf = pos_value_a*96768
                pos_sup = (pos_value_a + 2)*96768
                signal_original_anio_IN = signal_original_IN[pos_inf:pos_sup]
                signal_original_anio_OUT = signal_original_OUT[pos_inf:pos_sup]


            if(j == 193535):

                signal_reconstruida_anio_wavelet_Mbs_IN = [8 * float(x) / 1000000000 for x in signal_reconstruida_anio_wavelet_IN]
                signal_reconstruida_anio_wavelet_Mbs_OUT = [8 * float(x) / 1000000000 for x in signal_reconstruida_anio_wavelet_OUT]

                signal_reconstruida_anio_wavelet_Mbs_IN = np.array(signal_reconstruida_anio_wavelet_Mbs_IN)
                signal_reconstruida_anio_wavelet_Mbs_OUT = np.array(signal_reconstruida_anio_wavelet_Mbs_OUT)


		max_y_wavelet_in = round(np.amax(signal_reconstruida_anio_wavelet_Mbs_IN),3)
		max_y_wavelet_out = round(np.amax(signal_reconstruida_anio_wavelet_Mbs_OUT),3)
		mean_y_wavelet_in = round(np.mean(signal_reconstruida_anio_wavelet_Mbs_IN),3)
		mean_y_wavelet_out = round(np.mean(signal_reconstruida_anio_wavelet_Mbs_OUT),3)

               
                import datetime as dt
                import pytz
		tz = pytz.timezone('Europe/Madrid')

                fecha1 = dt.datetime.utcfromtimestamp(1171234800).replace(tzinfo=pytz.utc)
                dt = tz.normalize(fecha1.astimezone(tz))
                fecha1 = dt.strftime('%Y-%m-%d %H:%M')
                array_dates_year_1 = pd.date_range(fecha1, periods = 48372, freq = '300S')

  
	        import datetime as dt
                import pytz
		tz = pytz.timezone('Europe/Madrid')

                fecha2 = dt.datetime.utcfromtimestamp(1188770400).replace(tzinfo=pytz.utc)
                dt = tz.normalize(fecha2.astimezone(tz))
                fecha2 = dt.strftime('%Y-%m-%d %H:%M')
                array_dates_year_2 = pd.date_range(fecha2, periods = 94752, freq = '300S')
 
                import datetime as dt
                import pytz
		tz = pytz.timezone('Europe/Madrid')

                fecha3 = dt.datetime.utcfromtimestamp(1220220000).replace(tzinfo=pytz.utc)
                dt = tz.normalize(fecha3.astimezone(tz))
                fecha3 = dt.strftime('%Y-%m-%d %H:%M')
                array_dates_year_3 = pd.date_range(fecha3, periods = 50412, freq = '300S')

		array_dates_year_aux = array_dates_year_1.append(array_dates_year_2)
                array_dates_year = array_dates_year_aux.append(array_dates_year_3)

	        ts_year = pd.Series(signal_reconstruida_anio_wavelet_Mbs_IN, index=array_dates_year)

                w=2
	        h=len(signal_reconstruida_anio_wavelet_Mbs_IN)
                Matrix_r_wavelet_year = np.array([[0 for x in range(w)] for y in range(h)], dtype='f')

	        Matrix_r_wavelet_year[:,0] = signal_reconstruida_anio_wavelet_Mbs_IN.transpose()
                Matrix_r_wavelet_year[:,1] = signal_reconstruida_anio_wavelet_Mbs_OUT.transpose()

                # Wavelet
                fig = plt.figure()
                fig.suptitle('Input / Output Traffic Wavelet', fontsize=14, fontweight='bold')
	        df = pd.DataFrame(Matrix_r_wavelet_year, index=ts_year.index, columns=['Input Mbs','Output Mbs'])
                ax = df.plot(figsize=(10.5,3.5),color=['#00cc00', '#0000ff'], title='Yearly Wavelet Graph (Level 9, Sym17)',lw=1,colormap='jet');
                ax.fill_between(ts_year.index,0,Matrix_r_wavelet_year[:,0], facecolor='#00cc00')  

                ax.set_ylabel("Bits per Second")
                ax.legend().set_visible(False)
                labels_y_yearly = ax.yaxis.get_majorticklocs()
                new_labels_y_yearly = []
                for e in range(0,len(labels_y_yearly)):
                    labels_y_yearly[e]=format(labels_y_yearly[e], '.1f')
                    new_labels_y_yearly.append('%.1f G' %labels_y_yearly[e])   
                ax.yaxis.set_ticklabels(np.array(new_labels_y_yearly))
                
                monthsFmt=DateFormatter("%b 20%y")
                ax.xaxis.set_major_formatter(monthsFmt)
                ax.grid(b=True, which='both',color='k', linestyle=':',alpha=0.7)
                fig.patch.set_facecolor('#F3F3F3')
                plt.margins(x=0)
                plt.savefig('YearlyWavelet.png',facecolor=fig.get_facecolor())
	        plt.savefig('YearlyWavelet.eps',facecolor=fig.get_facecolor())

                fig.clear()

		generateHTMLfile (max_real_in, mean_real_in, max_real_out, mean_real_out, max_w_wavelet_in, mean_w_wavelet_in, max_w_wavelet_out, mean_w_wavelet_out, max_m_wavelet_in, mean_m_wavelet_in, max_m_wavelet_out, mean_m_wavelet_out, max_y_wavelet_in, mean_y_wavelet_in, max_y_wavelet_out, mean_y_wavelet_out)



matrix = np.loadtxt("input_signal.txt", dtype='i', delimiter=' ')

timestamps = []
signal_original_IN = []
signal_original_OUT = []


for i in range(0,len(matrix)):
    for j in range(0, 4):
        if j==0: 
            timestamps.append(matrix[i][j])
	if j==1: 
            signal_original_IN.append(matrix[i][j])
        if j==2:
            signal_original_OUT.append(matrix[i][j])


array_start_ts = [1171234800, 1188770400, 1220220000]
array_end_ts = [1185746100, 1217195700, 1248645300]

signal_auxiliar0 = []
signal_auxiliar1 = []
signal_auxiliar2 = []

for i in range(0, len(array_start_ts)):
    posicion_i = posTimestamp(matrix, array_start_ts[i])
    posicion_f = posTimestamp(matrix, array_end_ts[i])

    for j in range (posicion_i, posicion_f+1):
        signal_auxiliar0.append(timestamps[j])
        signal_auxiliar1.append(signal_original_IN[j])
        signal_auxiliar2.append(signal_original_OUT[j])


arrayTimestamps = np.array(signal_auxiliar0)
arrayTrafficIN = np.array(signal_auxiliar1)
arrayTrafficOUT = np.array(signal_auxiliar2)


# create matrix [2016x3] of floats:
w=3
h=len(arrayTimestamps)
Matrix = np.array([[0 for x in range(w)] for y in range(h)], dtype='f')


Matrix[:,0] = arrayTimestamps.transpose()
Matrix[:,1] = arrayTrafficIN .transpose()
Matrix[:,2] = arrayTrafficOUT.transpose()

bufferWavelet(Matrix)







