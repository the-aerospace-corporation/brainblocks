// =============================================================================
// _template.hpp
// =============================================================================
#ifndef TEMPLATE_HPP
#define TEMPLATE_HPP

#include "../block.hpp"
//#include "../block_input.hpp"
//#include "../block_memory.hpp"
//#include "../block_output.hpp"

//#include <vector>
//...

namespace BrainBlocks {

class Template final : public Block {

public:

    // Constructor
    Template(
        //const uint32_t param0,
        //const uint32_t param1,
        //...
    );

    // Overrided virtual functions
    void init() override;
    void save(const char* file) override;
    void load(const char* file) override;
    void clear() override;
    void step() override;
    void pull() override;
    void push() override;
    void encode() override;
    void decode() override;
    void learn() override;
    void store() override;
    uint32_t memory_usage() override;

    // Public functions
    // void public_function();
    //...

    // Block IO and Memory
    //BlockInput input;
    //BlockOutput output;
    //BlockMemory memory;
    //...

private:

    // Private functions
    //void private_function();
    //...

    // Private variables
    //uint32_t param0; // parameter 0
    //uint32_t param1; // parameter 1
    //uint32_t param2; // parameter 2
    //...
};

} // namespace BrainBlocks

#endif // TEMPLATE_HPP
