# Import libraries
using Test

# Import modules
include("../src/scratch_gru.jl")

# Unit test for the "gru_cell_forward" function
@testset "Test for gru_cell_forward" begin
      # Declare the test inputs
      x = [0.5 0.2 -0.1;
            0.4 0.1 -0.8;
            0.05 0.2 0.3;
            -0.3 0.2 0.3;
            0.3 0.2 0.2]
      input_size, n_samples = size(x)
      prev_h = [-0.5 1.0 -0.4;
            -0.8 -0.3 0.25]
      output_size = 2
      hidden_size = size(prev_h, 1)
      Wz = ones(hidden_size, hidden_size + input_size)
      Wr = ones(hidden_size, hidden_size + input_size)
      Wh = ones(hidden_size, hidden_size + input_size)
      Wy = ones(output_size, hidden_size)
      bz = ones(hidden_size, 1)
      br = ones(hidden_size, 1)
      bh = ones(hidden_size, 1)
      by = ones(output_size, 1)
      parameters = Dict("Wz" => Wz, "Wr" => Wr, "Wh" => Wh, "Wy" => Wy,
                        "bz" => bz, "br" => br, "bh" => bh, "by" => by)

      # Declare the expected value
      expected_next_h = [0.35346746 0.98875422 0.3219576;
                        0.2505706 0.89887428 0.53049145]
      expected_y_pred = [1.60403805 2.8876285 1.85244905;
                        1.60403805 2.8876285 1.85244905]

      # Evaluate the function
      next_h, y_pred, next_cache = gru_cell_forward(x, prev_h, parameters)
      @test round.(next_h, digits=8) == expected_next_h
      @test round.(y_pred, digits=8) == expected_y_pred
end


# Unit test for the "gru_forward" function
@testset "Test for gru_forward" begin
      # Declare the test inputs
      x = [0.99300052 0.7913093 0.3361028;
            0.11312647 0.36729219 0.76227817;
            0.84373686 0.99097526 0.88544693;
            0.16143223 0.17745618 0.15574817;;;
            
            0.702162 0.90674644 0.90636225;
            0.93672752 0.81236092 0.98507546;
            0.76851799 0.83472218 0.05869875;
            0.12252707 0.52074356 0.06808022;;;
            
            0.95409744 0.77087095 0.47433017;
            0.16305007 0.19577304 0.60276659;
            0.76963469 0.98756471 0.65530606;
            0.58611406 0.31873598 0.12422598;;;
            
            0.80381347 0.02690884 0.75007394;
            0.70237629 0.57737151 0.95403344;
            0.15675864 0.59819253 0.8423508;
            0.79587445 0.60350738 0.86909859;;;
            
            0.62412243 0.72277322 0.6556868;
            0.234298 0.02305273 0.17021789;
            0.58640138 0.45500247 0.84973528;
            0.96098881 0.60912028 0.92261943]
      input_size, n_samples, total_grus = size(x)
      ho = [-0.5 1.0 -0.4;
            -0.8 -0.3 0.25]
      hidden_size = size(ho, 1)
      output_size = 2
      Wz = ones(hidden_size, hidden_size + input_size)
      Wr = ones(hidden_size, hidden_size + input_size)
      Wh = ones(hidden_size, hidden_size + input_size)
      Wy = ones(output_size, hidden_size)
      bz = ones(hidden_size, 1)
      br = ones(hidden_size, 1)
      bh = ones(hidden_size, 1)
      by = ones(output_size, 1)

      parameters = Dict("Wz" => Wz, "Wr" => Wr, "Wh" => Wh, "Wy" => Wy,
                        "bz" => bz, "br" => br, "bh" => bh, "by" => by)

      # Declare the expected value
      expected_h = [0.75798745 0.99936022 0.92820247;
                  0.71584299 0.97659085 0.95933684;;;

                  0.998295 0.99998731 0.99936045;
                  0.99801399 0.99993378 0.99958927;;;

                  0.99995687 0.99994651 0.99987102;
                  0.99995569 0.99994624 0.99987279;;;
                  
                  0.99996308 0.99986241 0.99999441;
                  0.99996307 0.9998624 0.99999441;;;
                  
                  0.99995896 0.99986278 0.99997223;
                  0.99995896 0.99986278 0.99997223]
      expected_y = [2.47383044 2.97595107 2.88753931;
                  2.47383044 2.97595107 2.88753931;;;

                  2.99630899 2.99992109 2.99894971;
                  2.99630899 2.99992109 2.99894971;;;

                  2.99991256 2.99989276 2.99974382;
                  2.99991256 2.99989276 2.99974382;;;

                  2.99992615 2.99972481 2.99998882;
                  2.99992615 2.99972481 2.99998882;;;

                  2.99991792 2.99972556 2.99994446;
                  2.99991792 2.99972556 2.99994446]
      # Evaluate the function
      h, y, caches = gru_forward(x, ho, parameters)
      @test round.(h, digits=8) == expected_h
      @test round.(y, digits=8) == expected_y
end


# Unit test for the "gru_cell_backward" function
@testset "Test for gru_cell_backward" begin
      # Declare the test inputs
      x = [0.5 0.2 -0.1;
            0.4 0.1 -0.8;
            0.05 0.2 0.3;
            -0.3 0.2 0.3;
            0.3 0.2 0.2]
      input_size, n_samples = size(x)
      prev_h = [-0.5 1.0 -0.4;
            -0.8 -0.3 0.25]
      hidden_size = size(prev_h, 1)
      output_size = 2
      Wz = ones(hidden_size, hidden_size + input_size)
      Wr = ones(hidden_size, hidden_size + input_size)
      Wh = ones(hidden_size, hidden_size + input_size)
      Wy = ones(output_size, hidden_size)
      bz = ones(hidden_size, 1)
      br = ones(hidden_size, 1)
      bh = ones(hidden_size, 1)
      by = ones(output_size, 1)
      parameters = Dict("Wz" => Wz, "Wr" => Wr, "Wh" => Wh, "Wy" => Wy,
                        "bz" => bz, "br" => br, "bh" => bh, "by" => by)

      # Evaluate the function
      next_h, y_pred, next_cache = gru_cell_forward(x, prev_h, parameters)
      dh = 0.5 * ones(hidden_size, n_samples)
      gradients  = gru_cell_backward(dh, next_cache)

      @test size(gradients["dx"]) == (5, 3)
      @test round(gradients["dWz"][1,2],digits=8) == -0.27171002
      @test round(gradients["dbh"][2,1],digits=8) == 3.60201504
end


# Unit test for the "gru_backward" function
@testset "Test for gru_backward" begin
      # Declare the test inputs
      x = [0.99300052 0.7913093 0.3361028;
            0.11312647 0.36729219 0.76227817;
            0.84373686 0.99097526 0.88544693;
            0.16143223 0.17745618 0.15574817;;;
            
            0.702162 0.90674644 0.90636225;
            0.93672752 0.81236092 0.98507546;
            0.76851799 0.83472218 0.05869875;
            0.12252707 0.52074356 0.06808022;;;
            
            0.95409744 0.77087095 0.47433017;
            0.16305007 0.19577304 0.60276659;
            0.76963469 0.98756471 0.65530606;
            0.58611406 0.31873598 0.12422598;;;
            
            0.80381347 0.02690884 0.75007394;
            0.70237629 0.57737151 0.95403344;
            0.15675864 0.59819253 0.8423508;
            0.79587445 0.60350738 0.86909859;;;
            
            0.62412243 0.72277322 0.6556868;
            0.234298 0.02305273 0.17021789;
            0.58640138 0.45500247 0.84973528;
            0.96098881 0.60912028 0.92261943]
      input_size, n_samples, _ = size(x)
      ho = [-0.5 1.0 -0.4;
            -0.8 -0.3 0.25]
      hidden_size = size(ho, 1)
      output_size = 2
      Wz = ones(hidden_size, hidden_size + input_size)
      Wr = ones(hidden_size, hidden_size + input_size)
      Wh = ones(hidden_size, hidden_size + input_size)
      Wy = ones(output_size, hidden_size)
      bz = ones(hidden_size, 1)
      br = ones(hidden_size, 1)
      bh = ones(hidden_size, 1)
      by = ones(output_size, 1)
      parameters = Dict("Wz" => Wz, "Wr" => Wr, "Wh" => Wh, "Wy" => Wy,
                        "bz" => bz, "br" => br, "bh" => bh, "by" => by)

      # Evaluate the function
      h, y, caches = gru_forward(x, ho, parameters)
      dh = [0.03701893 -0.86856457 1.27804863;
            0.7674326 0.95530113 0.69208256;;;
            
            0.06154538 0.76764466 0.59373666;
            -1.34432403 -0.08361558 0.91675072;;;
            
            -1.34742525 0.36569409 0.80901497;
            -1.04883368 1.78671214 0.24059484;;;
            
            -0.44862616 0.85685679 -0.75122973;
            -0.58465207 -0.26490482 0.50344795;;;
            
            0.12878149 0.54781834 -0.10762447;
            -0.47655997 1.53565918 -0.02674601]
      gradients = gru_backward(dh, caches)

      @test size(gradients["dh_prev"]) == (2, 3, 5)
      @test round(gradients["dWr"][2,2], digits=3) == 0.229
end