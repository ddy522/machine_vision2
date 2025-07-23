using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WpfApp1
{
    public class Bom()
    {
            public string parent_code { get; set; }
            public string part_code { get; set; }
            public string part_name { get; set; }
            public int useage { get; set; }
            public int part_seq { get; set; }
    }
}
