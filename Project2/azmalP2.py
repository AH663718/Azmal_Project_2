import wx
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# --- Dataset and Feature Configuration ---
# Fetch the Heart Disease dataset (UCI ID 45)
heart_disease = fetch_ucirepo(id=45)

# Combine features (X) and target (y) into one DataFrame
# Note: X.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
# Note: y.columns = ['num'] (diagnosis of heart disease)
# num: 0=no disease, >0=disease
df_data = pd.concat([heart_disease.data.features, heart_disease.data.targets], axis=1)
df_data = df_data.dropna() # Remove rows with NaN values if they exist

FEATURE_X = "age"
FEATURE_Y = "chol"
TARGET_COL = "num"

# --- Application Class ---
class HeartVisualizerApp(wx.Frame):
    def __init__(self, parent=None, title="Heart Disease Dataset Visualizer"):
        super().__init__(parent, title=title, size=(820, 520))
        self.SetMinSize((760, 480))
        panel = wx.Panel(self)
        self.df = df_data # Store the pre-loaded DataFrame

        # Centered big button
        plot_btn = wx.Button(panel, label="Visualize Heart Data")
        font = plot_btn.GetFont()
        font.PointSize += 2
        plot_btn.SetFont(font)
        plot_btn.Bind(wx.EVT_BUTTON, self.on_plot)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.AddStretchSpacer(1)
        sizer.Add(plot_btn, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        sizer.AddStretchSpacer(2)
        panel.SetSizer(sizer)

        self.Centre()
        self.Show()

    def on_plot(self, _evt):
        try:
            # Plot — simple, readable scatter
            plt.figure(figsize=(7.5, 5.0))
            
            # The 'num' target can have values > 1, so we group by its
            # binary status: 0 (No Disease) vs >0 (Disease)
            self.df['Disease_Status'] = self.df[TARGET_COL].apply(lambda x: 'Disease' if x > 0 else 'No Disease')
            
            for status, sub in self.df.groupby('Disease_Status'):
                plt.scatter(
                    sub[FEATURE_X], 
                    sub[FEATURE_Y], 
                    s=40, 
                    alpha=0.85, 
                    label=status,
                    # Assign a color based on status for better visualization
                    color='red' if status == 'Disease' else 'blue'
                )

            # Titles and Labels
            title_text = f"Heart Disease: {FEATURE_X.title()} vs {FEATURE_Y.title()} (Grouped by Diagnosis)"
            plt.title(title_text)
            plt.xlabel(FEATURE_X.title())
            plt.ylabel(FEATURE_Y.title())
            
            # Legend and Layout
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.legend(title="Diagnosis Status", frameon=True)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            wx.MessageBox(f"Could not plot data:\n{e}", "Error", wx.OK | wx.ICON_ERROR)

if __name__ == "__main__":
    app = wx.App(False)
    HeartVisualizerApp()
    app.MainLoop()