using Blink
w = Window()
# API resembles the one from electron: https://github.com/electron/electron/blob/master/docs/api/browser-window.md
# Load html :3
loadurl(w, "file:///$(dirname(@__FILE__))/index.html")

# Can't get access to a specific element ?
content!(w, "div", "<div id=\"hey\" style=\"background-color:blue\"><p>wuuuuuuu</p></div>")

body!(w, """<button onclick='Blink.msg("press", "HELLO")'>go</button>""");

using PlotlyJS
p = plot([scatter(x=[1,2], y=[3,4]), scatter(y=[2, 1], x=[4,3])], Layout(title="My plot"))
update!(p, Layout(title="My asdas"))
