import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class Plot():
    FS = 128000 # sampling frequency in Hz

    @classmethod
    def radar_fmcw_signal(cls, complex_signal, name=False, size=(1000,500)):
        signal = np.asarray(complex_signal)
        amp = np.abs(signal)
        total_time = len(signal) / cls.FS  # Calculate total time duration
        time = np.linspace(0, total_time, len(signal))

        if not name:
            name = 'Raw Radar FMCW Signal'

        # Plotting the signal
        fig = go.Figure(layout=dict(
            title=dict(text=name),
            xaxis=dict(title='Time (s)', showgrid=True),
            yaxis=dict(title='Amplitude (V)', showgrid=True),
            width=size[0],
            height=size[1]
        ))
        fig.add_scatter(x=time, y=amp)

        return fig


    @classmethod
    def fft_filtered_sampling(cls, fft_p, fft_p_filt, fft_f, samples):
        sub = []
        for s in samples:
            sub += [f'Pulse {s}' for i in range(2)]

        fig = make_subplots(rows=len(samples), cols=2, vertical_spacing = 0.07, subplot_titles=sub)
        total_data = len(fft_p[0])
        
        for i, s in enumerate(samples, 1):
            # fft & filtered signal, half only which means the positive signal
            mag = 20 * np.log10(np.abs(fft_p[s]))[:total_data // 2]
            mag_filt = 20 * np.log10(np.abs(fft_p_filt[s]))[:total_data // 2]

            freq = fft_f[s][:total_data // 2]

            for f, m in zip(freq, mag):
                fig.add_trace(
                    go.Scatter(x=[f,f], y=[0,m], mode='lines', line=dict(color='blue')),
                    row=i, col=1
                )
                

            fig.add_trace(
                go.Scatter(x=freq, y=mag, mode='markers', marker=dict(color='red', size=5)),
                row=i, col=1
            )

            for f, m in zip(freq, mag_filt):
                fig.add_trace(
                    go.Scatter(x=[f,f], y=[0,m], mode='lines', line=dict(color='blue')),
                    row=i, col=2
                )
                

            fig.add_trace(
                go.Scatter(x=freq, y=mag_filt, mode='markers', marker=dict(color='red', size=5)),
                row=i, col=2
            )

            # fig.add_trace(
            #     go.Scatter(x=freq, y=mag, mode='lines', line=dict(color='blue')),
            #     row=i, col=1
            # )
            
            # fig.add_trace(
            #     go.Scatter(x=freq, y=mag_filt, mode='lines', line=dict(color='blue')),
            #     row=i, col=2
            # )

            fig.update_yaxes(title_text='Magnitude (dB)', row=i, col=1)
            fig.update_xaxes(title_text='Frequency (Hz)', row=i, col=1)
            fig.update_yaxes(title_text='Filtered Magnitude (dB)', row=i, col=2)
            fig.update_xaxes(title_text='Frequency (Hz)', row=i, col=2)
        
        fig.update_layout(height=1400, width=1200, showlegend=False, title_text='Sample Filtered FFT Signal Pulses')
        fig.show()

    
    @classmethod
    def fft_signal_sampling(cls, p, fft_p, fft_f, samples):
        sub = []
        for s in samples:
            sub += [f'Pulse {s}' for i in range(2)]
        fig = make_subplots(rows=len(samples), cols=2, vertical_spacing = 0.07, subplot_titles=sub)

        for i, s in enumerate(samples, 1):
            # raw signal
            sig = np.real(p[s])
            total_data = len(sig)

            # fft signal, half only which means the positive signal
            mag = 20 * np.log10(np.abs(fft_p[s]))[:total_data // 2]
            # mag = np.abs(fft_p[s])[:total_data // 2]
            freq = fft_f[s][:total_data // 2]

            total_time = total_data * 1000 / cls.FS # in ms
            time = np.linspace(0, total_time, len(sig))

            fig.add_trace(
                go.Scatter(x=time, y=sig, mode='lines', line=dict(color='red')),
                row=i, col=1
            )

            for f, m in zip(freq, mag):
                fig.add_trace(
                    go.Scatter(x=[f,f], y=[0,m], mode='lines', line=dict(color='blue')),
                    row=i, col=2
                )
                

            fig.add_trace(
                go.Scatter(x=freq, y=mag, mode='markers', marker=dict(color='red', size=5)),
                row=i, col=2
            )

            fig.update_yaxes(title_text='Amplitude (V)', row=i, col=1)
            fig.update_xaxes(title_text='Time (ms)', row=i, col=1)
            fig.update_yaxes(title_text='Magnitude (dB)', row=i, col=2)
            fig.update_xaxes(title_text='Frequency (Hz)', row=i, col=2)
    

        fig.update_layout(height=1400, width=1200, showlegend=False, title_text='Sample Radar FMCW & Windowed FFT Signal Pulses')
        fig.show()
        
    
    @classmethod
    def spectrogram(cls, v, t, Zxx, size= (6,6), name=False, img=False, **kwargs):        

        # Convert magnitude dB
        Zxx = np.abs(Zxx)
        # Zxx = 20 * np.log10(Zxx)

        if not name:
            name = 'Spectrogram' 


        # Plot spectrogram figure 
        plt.figure(figsize=size)
        plt.pcolormesh(t, v, Zxx, shading='gouraud', cmap='jet')
        
        if img == False:
            plt.ylabel('Velocity [m/s]')
            plt.xlabel('Time [sec]')
            plt.title(name)
            plt.colorbar(label='Magnitude (dB)')
            plt.show()

        else:
            plt.ioff()
            plt.axis('off')
            plt.gca().set_facecolor('white')
            plt.savefig(f"data/{kwargs['label']}/{kwargs['file']}.jpg", bbox_inches='tight', pad_inches=0)
            plt.close()

# import string
# ''.join([random.choice(string.ascii_letters) for x in range(8)])

# import string
# ''.join([random.choice(string.ascii_letters) for x in range(8)])