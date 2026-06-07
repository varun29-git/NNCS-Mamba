Quadrotor Basics
A quadrotor is a drone that moves in 3D space. To describe where it is and how it is moving, we need to track 12 values in total.
The first 6 describe its position and orientation. Position tells us where the drone is in space: x, y, and z. Orientation tells us how it is tilted or rotated: roll, pitch, and yaw. Roll means tilting left or right, pitch means tilting forward or backward, and yaw means turning left or right.
The next 6 describe how these quantities are changing over time. Position changes through linear velocity in x, y, and z. Orientation changes through angular velocity, which tells us how fast roll, pitch, and yaw are changing.
So the full state of the quadrotor is 12-dimensional: 6 values for position and orientation, and 6 values for their velocities.
This system is second-order. That means motion depends on acceleration, and acceleration depends on force. But instead of directly tracking acceleration, we track position and velocity, because those are enough to predict what happens next.
Virtual Environment and Time Scales
In the simulation, there are two different “clocks” running at different speeds.
The physics engine, PyBullet, updates at 1000 Hz, which means it updates the world every 0.001 seconds. It handles gravity, collisions, and physical forces.
The controller updates at 50 Hz, which means it makes a new decision every 0.02 seconds. This controller decides what motor speeds the drone should use.
This difference matters because the physics changes much faster than the controller. So by the time the controller acts, the world has already moved on a little.
Random Starting Position
The drone does not always start in the same perfect place. Instead, it starts with some randomness.
For example, its height may vary between 0.5 and 1.5 meters, and it may begin with a slight tilt or small initial velocity. This makes the task harder and more realistic.
The point is that the drone should not only work from one neat starting position. It should be able to recover from small disturbances too.
Goal of the Quadrotor
The task in this setup is simple to state but hard to achieve: keep the drone still in the air at a fixed point, usually 1 meter above the ground.
That means it should not move up or down, should not drift sideways, should not tilt, and should not spin. In other words, it should hover stably at one point in space.
This sounds easy, but the drone must constantly adjust its motors to fight gravity and stay balanced.
Why Hovering Is Difficult
Gravity is always pulling the drone downward. So the drone must generate exactly the right upward force. If it produces too little thrust, it falls. If it produces too much, it rises.
Also, position and orientation are tightly connected. If the drone tilts even a little, its thrust is no longer purely vertical. Some of that thrust becomes sideways force, which causes drift. So to control position, you must also control orientation.
The system is also naturally unstable. This is more like balancing a pencil on your finger than like a ball sitting in a bowl. If you stop correcting, it quickly gets worse.
There is also delay. The physics updates very fast, but the controller updates much more slowly. That means the control decisions are always slightly outdated.
In addition, the state of the drone may not be known perfectly. In real systems, and sometimes in simulation, sensors are noisy or imperfect, so the controller is working with estimated values rather than perfect truth.
Finally, the system has coupled dynamics. That means changing one thing affects several others. For example, increasing thrust changes height, but it can also affect stability. Correcting roll can change position. So nothing is independent.
Because of all this, the controller has to find a balance between accuracy and control effort. It should keep the drone stable without using ridiculous motor inputs.
What MPC Is
MPC stands for Model Predictive Control. The basic idea is simple: predict the future, choose the best control sequence, but apply only the first action. Then repeat everything again at the next step.
The loop is: observe the current state, use a physics model to predict future states, find the best motor inputs, apply the first one, and then re-plan again.
This is why it is called “receding horizon” control. The controller keeps moving forward one step at a time while always re-optimizing the future.
Prediction Horizon
In your setup, the prediction horizon is 20 steps. Since each step is 0.02 seconds, the controller looks about 0.4 seconds into the future.
A longer horizon lets the controller think more ahead, which can make it smarter. But it also makes the computation slower and harder. A shorter horizon is faster, but it can be too short-sighted.
Cost Function
MPC works by minimizing a cost function. The cost function tells the controller what “good behavior” means.
A common form is the sum of two parts: state error and control effort. State error measures how far the drone is from the target. Control effort measures how much motor input is being used.
In simple terms, the controller is trying to stay near the goal while not using unnecessary force.
Q and R Weights
The Q matrix decides how strongly the controller cares about state errors. Position has a weight of 5.0, while velocity has a weight of 0.1. That means the controller cares much more about staying at the right position than about making smooth motion.
The R matrix decides how much the controller cares about using motor force. In your setup, all motors have a weight of 0.1. That means the controller is allowed to use fairly strong motor commands if needed.
This combination creates aggressive correction: the drone tries hard to stay near the target, even if the motion becomes a little jerky.
What IPOPT Is
MPC needs to solve an optimization problem at every step. It must figure out which sequence of motor inputs gives the lowest cost while still obeying the physics and the motor limits.
That is not a simple problem. There are many variables, the dynamics are nonlinear, and there are constraints. So this becomes a nonlinear constrained optimization problem.
IPOPT is the solver that handles this. It searches for the best variable values that minimize the cost while respecting all constraints.
A useful way to think about it is this: imagine a landscape where height represents cost. IPOPT tries to find the lowest point. But the landscape is curved, and not every direction is allowed because of constraints.
IPOPT uses an interior-point method. That means it stays inside the valid region while moving toward better solutions. It uses gradients to know which way is downhill and Hessians to understand the curvature of the problem.
The important point is that IPOPT is what makes MPC practical. Without a strong optimizer, the controller would be too slow and too expensive to compute in real time.
Warm Start
Warm start means that instead of solving the optimization problem from scratch every time, you begin from the previous solution.
This works well because the system usually does not change drastically in 0.02 seconds. So the old solution is already close to the new best solution.
That makes the optimizer converge faster, which is crucial for real-time control.
Big Picture of MPC + IPOPT
At every timestep, MPC builds a math problem that asks: what sequence of motor commands is best for the next few steps? IPOPT solves that problem. Then only the first action is used. After that, the system measures the new state and repeats the whole process.
So the split is clear: MPC is the planning brain, the cost function defines what “good” means, Q and R define priorities, IPOPT solves the hard math, and warm start makes the whole thing fast enough to run online.
Forcing a Fixed Start State
The last function in your notes is a deterministic reset trick. Its job is to force the drone to start in one exact state instead of a random one.
It first checks that you provide exactly 12 values, because the full state has 12 dimensions. If you do not, the function fails immediately. That is good, because it prevents hidden bugs.
The main trick is very clever. Normally, the environment chooses a random value between a low and high bound. This function sets the low and high bounds to the same value. So the “random” choice can only return one number: the exact value you want.
This is not really removing randomness. It is forcing the system to behave deterministically.
That is useful for debugging and verification. If the drone fails in one weird situation, you want to reproduce that exact situation again and again. Without this trick, the failure state may never repeat. With this trick, you can replay the exact same start every time.
So this function is basically a precision reset tool. It lets you start from one known state instead of a random one.
Signal Temporal Logic and Robustness
Signal Temporal Logic, or STL, is used to describe how a system behaves over time. It is especially useful when you care not just about one moment, but about an entire trajectory.
The main idea is that instead of only asking “is this true or false?”, STL asks “how well is this condition satisfied?” That is called quantitative robustness. A positive robustness score means the condition is satisfied. A negative score means it is violated. The size of the score tells you how far the system is from success or failure.
Why Boolean logic is not enough
Boolean logic only gives two answers: true or false. That is too crude for control and optimization.
For example, if a drone must stay above a certain height, then 0.1 above the limit and 10 above the limit are both simply “true” in Boolean logic. But in practice, those two cases are very different. Robustness solves this problem by giving a continuous score instead of a yes/no answer. That makes it much more useful for learning, optimization, and falsification.
Temporal operators
STL has operators that reason across time, not just at one instant.
always means the condition must hold at every time step. The mathematical way to express this is by taking the minimum robustness value over the whole signal. If even one time step becomes unsafe, the whole condition becomes weak or violated.
eventually means the condition must become true at least once during the time interval. Mathematically, this is represented by the maximum robustness value over the signal. If the condition is strongly satisfied at any point, then the eventual condition succeeds.
suffix_always is a fast way to compute “always from now until the end.” It works by scanning backward through the signal and keeping track of the running minimum. That avoids slow repeated loops and makes temporal checks more efficient.
State safety
A key requirement in drone control is staying inside safe physical boundaries.
That means the drone should not cross the ceiling, should not hit the floor, and should remain within allowed position limits. To measure this, the system calculates how far the drone is from each boundary at every time step. The smallest of these distances is the most important one, because it is the closest threat.
If all the boundary margins are positive, the drone is safe. If any margin becomes negative, then a boundary has been crossed. Applying always to this safety signal means the drone must remain safe for the full trajectory, not just most of the time.
Input safety
The controller also has to respect motor limits.
A controller may calculate an action that is mathematically convenient but physically impossible. For example, it might ask for more thrust than the motors can actually produce. So the action must stay within the valid range of the environment.
To check this, the action is compared against the lower and upper bounds of the action space. If the action stays within the limits, the margin is positive. If it goes outside the limits, the margin becomes negative. Using always here means the control input must remain physically valid at all times.
Settling at the target
For stabilization, simply reaching the target once is not enough.
A drone can fly through the goal region at high speed and still not be considered stable. That is why the stronger condition eventually always is needed. It means the drone must eventually enter a settled state and then stay there.
A settled state usually means two things at the same time: the drone is close enough to the target position, and its speed is low enough. Then always checks that this settled condition continues to hold, and eventually checks that such a stable phase happens at least once.
This is the difference between “passing through the goal” and “actually stabilizing at the goal.” For hover control, stabilization is the real objective.
Why eventually always is stronger than eventually
eventually only asks whether the system reaches the target at some point.
eventually always asks something much stricter: whether the system reaches the target and then remains there for the rest of the time.
That is why eventually always is the right condition for a hovering drone. A fast fly-through may satisfy eventually, but it should not count as success if the drone does not stay stable afterward.
Final robustness score
All the important checks are combined into one final score using the minimum of the individual robustness values.
This behaves like logical AND. Every major requirement must be satisfied. If the drone is safe in most ways but fails one key condition, the final score becomes bad. That is exactly what you want, because even one serious violation means the whole trajectory is not acceptable.
So the robustness score acts as a single mathematical measure of whether the whole trajectory satisfies the safety and stabilization requirements. A positive value means success. A negative value means failure. That makes it extremely useful for optimization, testing, and counterexample search.
Big picture
The main idea is simple: turn drone behavior into a mathematical score over time.
That score checks whether the drone stays safe, respects actuator limits, reaches the target, and remains stable there. This is much more useful than plain true/false logic because it gives a smooth signal that can guide search and learning.
