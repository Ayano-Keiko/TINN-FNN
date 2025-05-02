// FeedForwardNerualNetwork.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define _CRT_SECURE_NO_WARNINGS

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <chrono>
#include <fstream>

#include "nn.h"
#include "includes/csv.h"


int main()
{
    int status = 0;

    std::vector<std::vector<double>> inputs;
    std::vector<double> targets;

    std::printf("DATA LOADING......\n");
    io::CSVReader<5> fp("./prepocess.csv");
    
    // read training value
    fp.read_header(io::ignore_extra_column, "altitude", "indicated_airspeed", "roll", "pitch", "pitch(t - 1)");
    // input features
    double altitude{};
    double indicated_airspeed{};
    double roll{};
    double previoud_pitch{};
    // target
    double pitch{};
    while (fp.read_row(altitude, indicated_airspeed, roll, pitch, previoud_pitch))
    {
        std::vector<double> row{};
        row.push_back(altitude);
        row.push_back(indicated_airspeed);
        row.push_back(roll);
        row.push_back(previoud_pitch);

        inputs.push_back(row);
        targets.push_back(pitch);

    }
    
    
    /*for (size_t i = 0; i < inputs.size(); ++i)
    {
        std::printf("inputs:\n");
        std::printf("altitude: %f\n", inputs[i][0]);
        std::printf("airspeed: %f\n", inputs[i][1]);
        std::printf("roll: %f\n", inputs[i][2]);
        std::printf("pitch(t - 1): %f\n", inputs[i][3]);
        std::printf("targets:\n");
        std::printf("pitch: %f\n\n", targets[i]);
    }*/
    

    // data meta info
    size_t size = inputs.size();
    size_t number_features = inputs[0].size();

    // printf("size: %zu\tfeature: %zu\n", size, number_features);

    // training
    std::printf("training......\n");
    FNN::NeuralNetwork net{ number_features };

    const auto start{std::chrono::steady_clock::now()};
    if ((status = net.train(inputs, targets, 50, 0.01)) == -1)
    {
        return status;
    }
    else {
        std::printf("Training Sucessfully!!\n");
    }
    const auto finish{ std::chrono::steady_clock::now() };

    const std::chrono::duration<double> elapsed_seconds{ finish - start };
    std::printf("Program time: %fs\n", elapsed_seconds.count() );

    std::vector<double> RMSE = net.getRMSE();

    
    FILE* log_fp = fopen("./logs.txt", "w");
    for (double loss : RMSE)
    {

        std::fprintf(log_fp, "%f\n", loss);
    }
    std::fclose(log_fp);
    
    
    return status;
}

