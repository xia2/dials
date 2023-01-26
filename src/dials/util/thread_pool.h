/*
 * thread_pool.h
 *
 *  Copyright (C) 2013 Diamond Light Source
 *
 *  Author: James Parkhurst
 *
 *  This code is distributed under the BSD license, a copy of which is
 *  included in the root directory of this package.
 */
#ifndef DIALS_ARRAY_FAMILY_THREAD_POOL_H
#define DIALS_ARRAY_FAMILY_THREAD_POOL_H

#include <atomic>

#include <boost/asio.hpp>
#include <boost/thread.hpp>

namespace dials { namespace util {

  /**
   * A class to implement a thread pool
   */
  class ThreadPool {
  public:
    /**
     * Instantiate with the number of required threads
     * @param N The number of threads
     */
    ThreadPool(std::size_t N) : work_(io_service_), started_(0), finished_(0) {
      for (std::size_t i = 0; i < N; ++i) {
        threads_.create_thread(
          boost::bind(&boost::asio::io_service::run, &io_service_));
      }
    }

    /**
     * Destroy the thread pool and join all threads
     */
    ~ThreadPool() {
      io_service_.stop();
      try {
        threads_.join_all();
      } catch (const std::exception &) {
        // pass
      }
    }

    /**
     * Post a function to the thread pool
     * @param function The function to call
     */
    template <typename Function>
    void post(Function function) {
      started_++;
      io_service_.post(FunctionRunner<Function>(function, finished_));
    }

    /**
     * Wait until all posted jobs have finished
     */
    void wait() {
      while (finished_ < started_)
        ;
    }

  protected:
    /**
     * A helper class to call the function increasing an atomic counter
     */
    template <typename Function>
    class FunctionRunner {
    public:
      /**
       * Create the helper class instance
       * @param function The function to call
       * @param counter The counter to increment
       */
      FunctionRunner(Function function, std::atomic<std::size_t> &counter)
          : function_(function), counter_(counter) {}

      /**
       * Call the function and increment the counter
       */
      void operator()() {
        function_();
        counter_++;
      }

    protected:
      Function function_;
      std::atomic<std::size_t> &counter_;
    };

    boost::asio::io_service io_service_;
    boost::asio::io_service::work work_;
    boost::thread_group threads_;
    std::size_t started_;
    std::atomic<std::size_t> finished_;
  };

}}  // namespace dials::util

#endif  // DIALS_ARRAY_FAMILY_THREAD_POOL_H
