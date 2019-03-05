using Blink
using PlotlyJS

w = Window()

# Add an event
handle(w, "press") do args
    @show args
end

# Run JS code in Window, trigger an event "Press" and pass the message
@js w Blink.msg("press", "Hello from JS"); # Trigger an event in JS

# Create a callback in Julia for when a event happens in the Window
handle(w, "event") do count, values, message
    println(x)
    # println(count)
    # println(values)
    # println(message)
end

# Trigger a event "event" and pass the arguments
@js w Blink.msg("event", [1, ['a','b'], "Hi"]);

# Creates an element in the window application which when clicked will trigger an event "press"
body!(w, """<button onclick='Blink.msg("press", "HELLO")'>go</button>""", async=false);

# BUT BEWARE!!!
# Note that you cannot make a synchronous call to javascript from within a julia callback, or you'll cause julia to hang:
# INSTEAD!!
# If you need to access the value of x, simply provide it when invoking the press handler
@js w x = 5
handle(w, "press") do args...
    x = args[1]
    # Increment x
    @js_ w (x = $x + 1)  # Note the _asynchronous_ call.
    println("New value: $x")
end

@js w Blink.msg("press", x)

# Julia webserver is implemented via Julia Tasks
# Julia code invoked from javascript will run in "parallel"  to the main julia code
# Particularly,
# - Tasks are coroutines, not threads (they don't really run in parallel)
# - Execution switch between code and coroutine's code whenever a piece of computation is interruptible
#
# !!! So, if your Blink callback handler performs uninterruptible work,
# it will fully occupy your CPU, preventing any other computation from occuring,
# and can potentially hang your computation.
# GOOD PRACTICE:
# So to allow for happy communication, all your computations should be
# interruptible, which you can achieve with calls such as yield, or sleep



# -----------------------------------------
# Async=false , blocks the code until the functions complete
# Important for sequential code, that depends on previous statement having completed
w = Window(async=false)
title(w, "Khepri-Optimizer :)")
body!(w, "Hello World", async=false)

# Setting content on a window:
# - content!
# - body!
# - load external url via loadurl, which replaces the content of the window
loadurl(w, "http://julialang.org")

# Load standalone HTML, CSS & JS files
# use load!, importhtml!, loadcss!, or loadjs!
load!(w, "ui/app.css")
load!(w, "ui/frameworks/jquery-3.1.1.js")
load!(w, "examples/visualization/index.html")

# Interaction between Julia and JS
# @js executes javascript directly
@js w Math.log(10)

# Invoke julia code frm javascript through callbacks
# Set up julia to handle the "press" message:
handle(w, "press") do args
  @show args
end
# Invoke the "press" message from javascript whenever this button is pressed:
body!(w, """<button onclick='Blink.msg("press", "HELLO")'>go</button>""");

# Window object
flashframe(w, true)
flashframe(w, false)

# Misc
opentools(w)
closetools(w)

tools(w)

@js w x = 5
@js_ w for i in 1:x console.log(i) end
