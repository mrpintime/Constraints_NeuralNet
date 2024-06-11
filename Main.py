import tensorflow as tf
import random
import numpy as np

list_of_conflicts = []

while len(list_of_conflicts) < 500:
    pairs_list = set()
    matrix = np.zeros((24, 24), dtype=int)
    
    while matrix.sum() < 40:
        num1 = np.random.choice(range(24))
        num2 = np.random.choice(range(24))
        
        if num1 == num2:
            continue
        
        pair = (num1, num2)
        if pair in pairs_list:
            continue
        
        pairs_list.add(pair)
        matrix[num1, num2] = 1

    if not any(np.array_equal(matrix, conflict) for conflict in list_of_conflicts):
        list_of_conflicts.append(matrix)

# data input is Conflict matrices
conflicts = np.array(list_of_conflicts)

def create_adjacent_mask(n_seats, seats_per_row, seats_per_col):
    adjacent_mask = np.zeros((n_seats, n_seats))
    for i in range(n_seats):
        if i % seats_per_row != 0:
            adjacent_mask[i, i-1] = 1
        if i % seats_per_row != seats_per_row-1:
            adjacent_mask[i, i+1] = 1
        if i >= seats_per_row:
            adjacent_mask[i, i-seats_per_row] = 1
        if i < n_seats-seats_per_row:
            adjacent_mask[i, i+seats_per_row] = 1
    return adjacent_mask

adjacent_mask = create_adjacent_mask(24,6,4)
# Define a function to calculate total conflicts for a given arrangement
def calculate_conflict(seating_arrangement, conflict_matrix):
    conflicts = 0
    ca_mul = tf.convert_to_tensor(conflict_matrix * adjacent_mask, tf.float64)
    conflicts = tf.reduce_sum(tf.matmul(tf.cast(seating_arrangement,tf.float64), ca_mul))
    # print(conflicts)
    return conflicts

# Custom loss function using the conflict calculation
def custom_loss(predicted_seating_arrangement, conflicts_tensor):
    loss = 0
    for i in range(predicted_seating_arrangement.shape[0]):
        conflict = tf.py_function(calculate_conflict,[predicted_seating_arrangement[i],conflicts_tensor[i]], tf.float64)
        # penalize the occurrences more than 1 
        occurrences = tf.reduce_sum(predicted_seating_arrangement[i], axis=0)

        # Calculate the number of repetitive elements
        repetitive_elements = tf.reduce_sum(tf.abs(occurrences - 1))/24

        loss += tf.cast(repetitive_elements, tf.float64) + conflict
    return tf.cast(loss, tf.float64) / tf.cast(predicted_seating_arrangement.shape[0], tf.float64)

conflicts_tensor = tf.convert_to_tensor(conflicts, tf.float64)

# Define the neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(conflicts_tensor.shape[1:]),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(24*24, activation='softmax'),
    tf.keras.layers.Reshape((24,24))
])

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Training loop
for epoch in range(2):
    with tf.GradientTape() as tape:
        predicted_seating_arrangement = model(conflicts_tensor, training=True)
        # print(predicted_seating_arrangement.shape)
        # Calculate the loss
        loss = custom_loss(predicted_seating_arrangement, conflicts_tensor)
    # Calculate the gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # Update the weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    # Print the loss
    print(f'Epoch: {epoch}, Loss: {loss.numpy()}')






z = model(conflicts_tensor)

# convert first prediction to seating arrangement of 6X4
q = tf.reshape(tf.argmax(z[0]), (6,4))
print(np.unique(q).size, q)
