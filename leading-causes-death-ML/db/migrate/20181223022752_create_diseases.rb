class CreateDiseases < ActiveRecord::Migration[5.2]
  def change
    create_table :diseases do |t|
      t.integer :year
      t.string :leading_cause
      t.string :sex
      t.string :race_ethnicity
      t.integer :deaths
      t.integer :death_rate
      t.integer :age_adjusted_death_rate

      t.timestamps
    end
  end
end
