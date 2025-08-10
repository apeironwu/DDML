# import holoviews as hv
# import numpy as np
# from bokeh.io import export_svgs
# import selenium 

# #set the backend and the renderer
# hv.extension('bokeh')
# br = hv.renderer('bokeh')

# #set the data
# np.random.seed(37)
# groups = [chr(65+g) for g in np.random.randint(0, 3, 200)]

# #plot 
# violin = hv.Violin((groups, np.random.randint(0, 5, 200), np.random.randn(200)),
#           ['Group', 'Category'], 'Value')

# #export the plot
# plot = br.get_plot(violin )
# plot = plot.state
# plot.output_backend='svg'
# export_svgs(plot, filename="Violin.svg")



from selenium import webdriver

driver = webdriver.Firefox()
driver.get('http://www.google.com/');
