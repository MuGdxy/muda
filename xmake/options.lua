-- *******************
--
-- development options
--
-- *******************
option("dev")
    set_default(false)
    set_showmenu(true)
    set_category("root menu/dev")
    set_description("build muda example, playground and test. if you're the developer, you could enable this option.")
option_end()

function option_dev_related() 
    add_deps("dev")
    after_check(function(option)
        if option:dep("dev"):enabled() then
            option:enable(true)
        end
    end)
end

option("example")
    set_default(true)
    set_showmenu(true)
    set_category("root menu")
    set_description("build muda examples. if you want to see how to use muda, you could enable this option.")
    option_dev_related()
option_end()

option("test")
    set_default(false)
    set_showmenu(true)
    set_description("build muda test. if you're the developer, you should enable this option.")
    set_category("root menu/dev")
    option_dev_related()
option_end()

option("playground")
    set_default(false)
    set_showmenu(true)
    set_description("build muda playground. if you're the developer, you could enable this option.")
    set_category("root menu/dev")
    option_dev_related()
option_end()

-- *******************
--
-- config macro options
--
-- *******************
option("with_check")
    set_default(true)
    set_showmenu(true)
    set_description("turn on all muda runtime check.")
    set_category("root menu/config")
option_end()

option("with_compute_graph")
    set_default(false)
    set_showmenu(true)
    set_description("turn on muda compute graph.")
    set_category("root menu/config")
option_end()