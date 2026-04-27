import pulp
import pandas as pd

def main():
    print("=== CODTECH Internship: Task 4 - Optimization Model ===")
    print("Solving a Production Planning Problem using PuLP\n")
    
    # --- Problem Setup ---
    # A company produces two products: Product A and Product B.
    # Profit per unit: Product A = $40, Product B = $30
    # Constraints:
    # Machine 1 hours available: 40 hours/week
    # Machine 2 hours available: 60 hours/week
    # Production Requirements:
    # Product A requires 2 hours on Machine 1 and 1 hour on Machine 2.
    # Product B requires 1 hour on Machine 1 and 3 hours on Machine 2.
    # Goal: Maximize total profit.
    
    # 1. Initialize the Optimization Model
    model = pulp.LpProblem("Profit_Maximization_Problem", pulp.LpMaximize)
    
    # 2. Define Decision Variables
    # Variables are the number of units to produce for A and B. 
    # They must be integers and cannot be negative.
    units_a = pulp.LpVariable('Units_Product_A', lowBound=0, cat='Integer')
    units_b = pulp.LpVariable('Units_Product_B', lowBound=0, cat='Integer')
    
    # 3. Define Objective Function
    # Maximize Profit = 40 * A + 30 * B
    model += 40 * units_a + 30 * units_b, "Total_Profit"
    
    # 4. Define Constraints
    # Machine 1 Constraint: 2*A + 1*B <= 40
    model += 2 * units_a + 1 * units_b <= 40, "Machine_1_Constraint"
    
    # Machine 2 Constraint: 1*A + 3*B <= 60
    model += 1 * units_a + 3 * units_b <= 60, "Machine_2_Constraint"
    
    print("Problem Formulation:")
    print(model)
    print("-" * 40)
    
    # 5. Solve the Model
    print("Solving the model...")
    model.solve()
    
    # 6. Extract and Display Results
    status = pulp.LpStatus[model.status]
    print(f"Status: {status}\n")
    
    if status == 'Optimal':
        opt_a = units_a.varValue
        opt_b = units_b.varValue
        max_profit = pulp.value(model.objective)
        
        # Create a summary table
        results_data = {
            'Product': ['Product A', 'Product B'],
            'Optimal Units to Produce': [opt_a, opt_b],
            'Profit Per Unit': ['$40', '$30'],
            'Total Profit Contribution': [f"${opt_a * 40}", f"${opt_b * 30}"]
        }
        df_results = pd.DataFrame(results_data)
        
        print("=== Optimization Results ===")
        print(df_results.to_string(index=False))
        print(f"\nTotal Maximum Profit: ${max_profit}")
        
        # Insights
        print("\n=== Business Insights ===")
        print(f"To maximize profit, the factory should produce {int(opt_a)} units of Product A")
        print(f"and {int(opt_b)} units of Product B each week.")
        
        # Calculate Machine Usage
        m1_usage = 2 * opt_a + 1 * opt_b
        m2_usage = 1 * opt_a + 3 * opt_b
        
        print(f"\nResource Utilization:")
        print(f"Machine 1: {m1_usage} out of 40 hours used ({(m1_usage/40)*100:.1f}%)")
        print(f"Machine 2: {m2_usage} out of 60 hours used ({(m2_usage/60)*100:.1f}%)")
        
        if m1_usage == 40:
            print("Insight: Machine 1 is fully utilized (bottleneck). Increasing its capacity would allow for more profit.")
        if m2_usage == 60:
            print("Insight: Machine 2 is fully utilized (bottleneck). Increasing its capacity would allow for more profit.")
            
    else:
        print("An optimal solution could not be found.")

    print("\n=== Task 4 Completed Successfully ===")

if __name__ == "__main__":
    main()
