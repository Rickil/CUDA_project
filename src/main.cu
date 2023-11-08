#include "image.hh"
#include "pipeline.hh"
#include "fix_cpu.cuh"
#include "fix_gpu.cuh"
#include "fix_gpu_perfect.cuh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto& dir_entry : recursive_directory_iterator("../images"))
        filepaths.emplace_back(dir_entry.path());

    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

    std::cout << "START\n";
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    cudaEventRecord(gpu_start);

    // CPU timer
    clock_t cpu_start, cpu_end;
    cpu_start = clock();

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        // TODO: Make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline:
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speed up this process, of course (wait for the last class)

        // i = 20;
        // printf("image: %d\n", i);
        images[i] = pipeline.get_image(i);
        fix_image_gpu(images[i]);
        /*images[i] = pipeline.get_image(i);
        fix_image_cpu(images[i]);*/
        // break;
    }

    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);

    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, gpu_start, gpu_stop);

    cpu_end = clock();
    float cpu_milliseconds = 1000.0f * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    std::cout << "Done with compute, starting stats" << std::endl;
    std::cout << "GPU Time: " << gpu_milliseconds << " milliseconds" << std::endl;
    std::cout << "CPU Time: " << cpu_milliseconds << " milliseconds" << std::endl;
    std::cout << "GPU vs CPU Time: " << gpu_milliseconds << " ms (GPU) vs " << cpu_milliseconds << " ms (CPU)" << std::endl;


    /*double seconds = milliseconds / 1000.0;
    double fps = images.size() / seconds;

    std::cout << "FPS: " << fps << std::endl;*/

    cudaEventDestroy(gpu_start);
    cudaEventDestroy(gpu_stop);


    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // TODO : make it GPU compatible (aka faster)
    // You can use multiple CPU threads for your GPU version using openmp or not
    // Up to you :)
    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto& image = images[i];
        const int image_size = image.width * image.height;
        image.to_sort.total = std::reduce(image.buffer, image.buffer + image_size, 0);
    }

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
    {
        return images[n++].to_sort;
    });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
        return a.total < b.total;
    });

    // TODO : Test here that you have the same results
    // You can compare visually and should compare image vectors values and "total" values
    // If you did the sorting, check that the ids are in the same order
    for (int i = 0; i < nb_images; ++i)
    {
        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    // Cleaning
    // TODO : Don't forget to update this if you change allocation style
    for (int i = 0; i < nb_images; ++i)
        free(images[i].buffer);

    return 0;
}
